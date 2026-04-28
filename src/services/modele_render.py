from __future__ import annotations

from datetime import date as _date, datetime
from decimal import Decimal, InvalidOperation
from html import escape as _escape_html
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.models.client import Client
from src.models.modele_document import ModeleDocument


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text_value = str(value).strip()
    return text_value


def _fmt_date(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, (datetime, _date)):
        try:
            return value.strftime("%d/%m/%Y")
        except Exception:
            return str(value)
    return _safe_text(value)


def _fmt_money(value: Any, *, decimals: int = 0) -> str:
    if value in (None, ""):
        return ""
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return _safe_text(value)
    quantized = f"{amount:,.{decimals}f}"
    return quantized.replace(",", " ") + " €"


def _join_non_empty(values: list[str], sep: str = " - ") -> str:
    filtered = [v for v in (str(x).strip() for x in values) if v]
    return sep.join(filtered)


def _set_many(ctx: dict[str, str], value: Any, *keys: str) -> None:
    text_value = _safe_text(value)
    for key in keys:
        ctx[key] = text_value


def _summary_rows(rows: list[dict[str, Any]], label_key: str, amount_key: str, *, limit: int = 5) -> str:
    parts: list[str] = []
    for row in rows[:limit]:
        label = _safe_text(row.get(label_key)) or "Non renseigné"
        amount = row.get(amount_key)
        amount_str = _fmt_money(amount, decimals=0) if amount not in (None, "") else ""
        if amount_str:
            parts.append(f"{label}: {amount_str}")
        else:
            parts.append(label)
    return " ; ".join(parts)


def _select_one_row(db: Session, sql: str, params: dict[str, Any]) -> dict[str, Any] | None:
    try:
        row = db.execute(text(sql), params).fetchone()
    except Exception:
        return None
    if not row:
        return None
    mapping = row._mapping if hasattr(row, "_mapping") else None
    if mapping is None:
        try:
            return dict(row)
        except Exception:
            return None
    return dict(mapping)


def build_client_placeholders(db: Session, client_id: int) -> dict[str, str]:
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        return {}

    ctx: dict[str, str] = {}

    # Identité simple
    _set_many(ctx, getattr(client, "prenom", None), "prenom_client", "client_prenom", "souscripteur1_prenom_usage", "souscripteur1_prenom_etat_civil")
    _set_many(ctx, getattr(client, "nom", None), "nom_client", "client_nom", "souscripteur1_nom")
    _set_many(ctx, getattr(client, "telephone", None), "telephone_client", "client_telephone")
    _set_many(ctx, getattr(client, "email", None), "email_client", "client_email")
    _set_many(ctx, getattr(client, "adresse_postale", None), "adresse_postale", "client_adresse_postale")
    _set_many(ctx, getattr(client, "id", None), "reference_dossier", "client_id", "reference_client")

    # Coordonnées de base
    full_name = _join_non_empty([ctx.get("prenom_client"), ctx.get("nom_client")], sep=" ")
    _set_many(ctx, full_name, "client_fullname", "nom_complet", "nom_client_complet")

    # État civil
    etat_civil = _select_one_row(
        db,
        """
        SELECT civilite,
               date_naissance,
               lieu_naissance,
               nationalite,
               situation_familiale,
               profession,
               commentaire
        FROM etat_civil_client
        WHERE id_client = :cid
        ORDER BY id DESC
        LIMIT 1
        """,
        {"cid": client_id},
    ) or {}

    civilite = _safe_text(etat_civil.get("civilite"))
    date_naissance = _fmt_date(etat_civil.get("date_naissance"))
    lieu_naissance = _safe_text(etat_civil.get("lieu_naissance"))
    nationalite = _safe_text(etat_civil.get("nationalite"))
    situation_familiale = _safe_text(etat_civil.get("situation_familiale"))
    profession = _safe_text(etat_civil.get("profession"))

    _set_many(ctx, civilite, "civilite_client", "client_civilite")
    _set_many(ctx, date_naissance, "date_naissance", "client_date_naissance", "souscripteur1_date_naissance")
    _set_many(ctx, lieu_naissance, "lieu_naissance", "client_lieu_naissance")
    _set_many(ctx, nationalite, "nationalite_client", "client_nationalite")
    _set_many(ctx, situation_familiale, "situation_familiale")
    _set_many(ctx, profession, "profession_client")

    etat_civil_label = _join_non_empty([civilite, situation_familiale], sep=" - ")
    if not etat_civil_label:
        etat_civil_label = civilite or situation_familiale
    _set_many(ctx, etat_civil_label, "etat_civil")

    # Situation matrimoniale / régime matrimonial
    matr = _select_one_row(
        db,
        """
        SELECT m.*, sm.libelle AS situation_libelle, sc.libelle AS convention_libelle
        FROM KYC_Client_Situation_Matrimoniale m
        LEFT JOIN ref_situation_matrimoniale sm ON sm.id = m.situation_id
        LEFT JOIN ref_situation_matrimoniale_convention sc ON sc.id = m.convention_id
        WHERE m.client_id = :cid
        ORDER BY (m.date_saisie IS NULL), m.date_saisie DESC, m.id DESC
        LIMIT 1
        """,
        {"cid": client_id},
    ) or {}

    situation_libelle = _safe_text(matr.get("situation_libelle"))
    convention_libelle = _safe_text(matr.get("convention_libelle"))
    _set_many(ctx, situation_libelle, "situation_matrimoniale")
    _set_many(ctx, convention_libelle, "regime_matrimonial")

    # Patrimoine / revenus
    synthese = _select_one_row(
        db,
        """
        SELECT total_revenus,
               total_charges,
               total_actif,
               total_passif,
               commentaire
        FROM KYC_Client_Synthese
        WHERE client_id = :cid
        ORDER BY date_saisie DESC, id DESC
        LIMIT 1
        """,
        {"cid": client_id},
    ) or {}

    revenus_total = synthese.get("total_revenus")
    charges_total = synthese.get("total_charges")
    actif_total = synthese.get("total_actif")
    passif_total = synthese.get("total_passif")

    try:
        patrimoine_net = (Decimal(str(actif_total or 0)) - Decimal(str(passif_total or 0)))
    except Exception:
        patrimoine_net = None

    revenus_annuels_str = _fmt_money(revenus_total)
    charges_annuelles_str = _fmt_money(charges_total)
    actif_total_str = _fmt_money(actif_total)
    passif_total_str = _fmt_money(passif_total)
    patrimoine_net_str = _fmt_money(patrimoine_net) if patrimoine_net is not None else ""
    patrimoine_revenus = _join_non_empty(
        [
            f"Revenus annuels: {revenus_annuels_str}" if revenus_annuels_str else "",
            f"Charges annuelles: {charges_annuelles_str}" if charges_annuelles_str else "",
            f"Actifs: {actif_total_str}" if actif_total_str else "",
            f"Passifs: {passif_total_str}" if passif_total_str else "",
            f"Patrimoine net: {patrimoine_net_str}" if patrimoine_net_str else "",
        ],
        sep=" | ",
    )
    _set_many(ctx, revenus_annuels_str, "revenus_annuels")
    _set_many(ctx, actif_total_str, "actifs_total", "patrimoine_total")
    _set_many(ctx, passif_total_str, "passifs_total")
    _set_many(ctx, patrimoine_net_str, "patrimoine_net")
    _set_many(ctx, patrimoine_revenus, "patrimoine_revenus")

    # Agrégats détaillés patrimoine / revenus / charges
    actifs_rows: list[dict[str, Any]] = []
    try:
        rows = db.execute(
            text(
                """
                SELECT a.id,
                       a.type_actif_id,
                       COALESCE(t.libelle, 'Non renseigné') AS type_libelle,
                       a.description,
                       a.valeur,
                       a.date_saisie,
                       a.date_expiration
                FROM KYC_Client_Actif a
                LEFT JOIN ref_type_actif t ON t.id = a.type_actif_id
                WHERE a.client_id = :cid
                ORDER BY (a.date_saisie IS NULL), a.date_saisie DESC, a.id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        for row in rows or []:
            data = row._mapping if hasattr(row, "_mapping") else {}
            actifs_rows.append(
                {
                    "type_libelle": data.get("type_libelle"),
                    "description": data.get("description"),
                    "valeur": data.get("valeur"),
                }
            )
    except Exception:
        actifs_rows = []

    passifs_rows: list[dict[str, Any]] = []
    try:
        rows = db.execute(
            text(
                """
                SELECT p.id,
                       p.type_passif_id,
                       COALESCE(t.libelle, 'Non renseigné') AS type_libelle,
                       p.description,
                       p.montant_rest_du,
                       p.date_saisie,
                       p.date_expiration
                FROM KYC_Client_Passif p
                LEFT JOIN ref_type_passif t ON t.id = p.type_passif_id
                WHERE p.client_id = :cid
                ORDER BY (p.date_saisie IS NULL), p.date_saisie DESC, p.id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        for row in rows or []:
            data = row._mapping if hasattr(row, "_mapping") else {}
            passifs_rows.append(
                {
                    "type_libelle": data.get("type_libelle"),
                    "description": data.get("description"),
                    "montant": data.get("montant_rest_du"),
                }
            )
    except Exception:
        passifs_rows = []

    revenus_rows: list[dict[str, Any]] = []
    try:
        rows = db.execute(
            text(
                """
                SELECT r.id,
                       r.type_revenu_id,
                       COALESCE(t.libelle, 'Non renseigné') AS type_libelle,
                       r.montant_annuel,
                       r.date_saisie,
                       r.date_expiration
                FROM KYC_Client_Revenus r
                LEFT JOIN ref_type_revenu t ON t.id = r.type_revenu_id
                WHERE r.client_id = :cid
                ORDER BY (r.date_saisie IS NULL), r.date_saisie DESC, r.id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        for row in rows or []:
            data = row._mapping if hasattr(row, "_mapping") else {}
            revenus_rows.append(
                {
                    "type_libelle": data.get("type_libelle"),
                    "montant": data.get("montant_annuel"),
                }
            )
    except Exception:
        revenus_rows = []

    charges_rows: list[dict[str, Any]] = []
    try:
        rows = db.execute(
            text(
                """
                SELECT c.id,
                       c.type_charge_id,
                       COALESCE(t.libelle, 'Non renseigné') AS type_libelle,
                       c.montant_annuel,
                       c.date_saisie,
                       c.date_expiration
                FROM KYC_Client_Charges c
                LEFT JOIN ref_type_charge t ON t.id = c.type_charge_id
                WHERE c.client_id = :cid
                ORDER BY (c.date_saisie IS NULL), c.date_saisie DESC, c.id DESC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        for row in rows or []:
            data = row._mapping if hasattr(row, "_mapping") else {}
            charges_rows.append(
                {
                    "type_libelle": data.get("type_libelle"),
                    "montant": data.get("montant_annuel"),
                }
            )
    except Exception:
        charges_rows = []

    _set_many(
        ctx,
        _summary_rows(actifs_rows, "type_libelle", "valeur"),
        "actifs_financiers",
    )
    _set_many(
        ctx,
        _summary_rows(passifs_rows, "type_libelle", "montant"),
        "passifs",
    )

    revenus_summary = _summary_rows(revenus_rows, "type_libelle", "montant")
    charges_summary = _summary_rows(charges_rows, "type_libelle", "montant")
    if revenus_summary:
        _set_many(ctx, revenus_summary, "revenus_details")
    if charges_summary:
        _set_many(ctx, charges_summary, "charges_details")

    # Connaissance / risque
    risque = _select_one_row(
        db,
        """
        SELECT
          r.libelle AS niveau_risque,
          k.niveau_id AS niveau_id,
          k.duree AS horizon_placement,
          k.experience,
          k.connaissance,
          k.commentaire,
          k.confirmation_client
        FROM KYC_Client_Risque k
        JOIN ref_niveau_risque r ON k.niveau_id = r.id
        WHERE k.client_id = :cid
        ORDER BY k.date_saisie DESC, k.id DESC
        LIMIT 1
        """,
        {"cid": client_id},
    ) or {}

    niveau_risque = _safe_text(risque.get("niveau_risque"))
    horizon_placement = _safe_text(risque.get("horizon_placement"))
    experience = _safe_text(risque.get("experience"))
    connaissance = _safe_text(risque.get("connaissance"))
    connaissance_financiere = _join_non_empty([connaissance, experience], sep=" - ")
    if not connaissance_financiere:
        connaissance_financiere = niveau_risque

    _set_many(ctx, niveau_risque, "profil_risque", "niveau_risque")
    _set_many(ctx, horizon_placement, "horizon_placement")
    _set_many(ctx, connaissance_financiere, "connaissance_financiere")
    _set_many(ctx, experience, "experience_investissement")
    _set_many(ctx, _safe_text(risque.get("commentaire")), "commentaire_risque")

    # Objectifs
    objectifs_labels: list[str] = []
    try:
        rows = db.execute(
            text(
                """
                SELECT o.id,
                       ro.libelle AS libelle,
                       o.horizon_investissement,
                       o.commentaire
                FROM KYC_Client_Objectifs o
                LEFT JOIN ref_objectif ro ON ro.id = o.objectif_id
                WHERE o.client_id = :cid
                ORDER BY o.id ASC
                """
            ),
            {"cid": client_id},
        ).fetchall()
        for row in rows or []:
            data = row._mapping if hasattr(row, "_mapping") else {}
            label = _safe_text(data.get("libelle"))
            if label:
                objectifs_labels.append(label)
    except Exception:
        objectifs_labels = []
    _set_many(ctx, " ; ".join(objectifs_labels), "objectif_placement")

    # ESG
    esg = _select_one_row(
        db,
        """
        SELECT id,
               env_importance,
               env_ges_reduc,
               soc_droits_humains,
               soc_parite,
               gov_transparence,
               gov_controle_ethique
        FROM esg_questionnaire
        WHERE client_ref = :ref
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        {"ref": str(client_id)},
    ) or {}

    esg_id = esg.get("id")
    esg_summary_bits: list[str] = []
    for key, label in (
        ("env_importance", "Environnement"),
        ("env_ges_reduc", "Réduction GES"),
        ("soc_droits_humains", "Droits humains"),
        ("soc_parite", "Parité"),
        ("gov_transparence", "Transparence"),
        ("gov_controle_ethique", "Éthique"),
    ):
        val = _safe_text(esg.get(key))
        if val:
            esg_summary_bits.append(f"{label}: {val}")

    esg_exclusions: list[str] = []
    esg_indicators: list[str] = []
    if esg_id is not None:
        try:
            rows = db.execute(
                text(
                    """
                    SELECT COALESCE(o.label, CONCAT('Option #', qe.option_id)) AS label
                    FROM esg_questionnaire_exclusion qe
                    LEFT JOIN esg_exclusion_option o ON o.id = qe.option_id
                    WHERE qe.questionnaire_id = :q
                    ORDER BY o.label
                    """
                ),
                {"q": int(esg_id)},
            ).fetchall()
            esg_exclusions = [str(r[0]) for r in rows if str(r[0]).strip()]
        except Exception:
            esg_exclusions = []
        try:
            rows = db.execute(
                text(
                    """
                    SELECT COALESCE(o.label, CONCAT('Option #', qi.option_id)) AS label
                    FROM esg_questionnaire_indicator qi
                    LEFT JOIN esg_indicator_option o ON o.id = qi.option_id
                    WHERE qi.questionnaire_id = :q
                    ORDER BY o.label
                    """
                ),
                {"q": int(esg_id)},
            ).fetchall()
            esg_indicators = [str(r[0]) for r in rows if str(r[0]).strip()]
        except Exception:
            esg_indicators = []

    _set_many(ctx, " ; ".join(esg_summary_bits), "sensibilite_esg")
    _set_many(ctx, " ; ".join(esg_indicators), "pref_esg")
    _set_many(ctx, " ; ".join(esg_exclusions), "exclusions_esg")

    # Conseiller / commercial
    conseiller_nom = ""
    conseiller_email = ""
    commercial_id = getattr(client, "commercial_id", None)
    if commercial_id is not None:
        try:
            row = db.execute(
                text("SELECT id, prenom, nom, mail FROM administration_RH WHERE id = :rid LIMIT 1"),
                {"rid": int(commercial_id)},
            ).fetchone()
            if row:
                data = row._mapping if hasattr(row, "_mapping") else {}
                conseiller_nom = _join_non_empty([data.get("prenom"), data.get("nom")], sep=" ")
                conseiller_email = _safe_text(data.get("mail"))
        except Exception:
            conseiller_nom = ""
            conseiller_email = ""
    _set_many(ctx, conseiller_nom, "conseiller_nom", "commercial_nom")
    _set_many(ctx, conseiller_email, "conseiller_email", "commercial_email")

    # Dates / références système
    today = _date.today()
    now = datetime.utcnow()
    _set_many(ctx, today.strftime("%d/%m/%Y"), "date_document")
    _set_many(ctx, now.strftime("%d/%m/%Y %H:%M"), "date_generation")
    _set_many(ctx, str(getattr(client, "id", "") or ""), "client_id_str")

    # Compatibilité / alias usuels
    if ctx.get("etat_civil") and not ctx.get("client_etat_civil"):
        _set_many(ctx, ctx["etat_civil"], "client_etat_civil")
    if ctx.get("patrimoine_revenus") and not ctx.get("patrimoine_revenus_summary"):
        _set_many(ctx, ctx["patrimoine_revenus"], "patrimoine_revenus_summary")
    if ctx.get("client_fullname") and not ctx.get("fullname"):
        _set_many(ctx, ctx["client_fullname"], "fullname")

    return ctx


def _build_render_context(db: Session, data: dict[str, Any] | None) -> dict[str, str]:
    ctx: dict[str, str] = {}
    if not data:
        return ctx

    raw = dict(data)
    nested_placeholders = raw.pop("placeholders", None)
    client_id = raw.pop("client_id", None)
    if client_id not in (None, ""):
        try:
            ctx.update(build_client_placeholders(db, int(client_id)))
        except Exception:
            pass
    if isinstance(nested_placeholders, dict):
        for key, value in nested_placeholders.items():
            ctx[str(key)] = _safe_text(value)
    for key, value in raw.items():
        if key == "client_id":
            continue
        ctx[str(key)] = _safe_text(value)
    return ctx


def render_modele(db: Session, modele_id: int, data: dict | None) -> dict[str, str]:
    m = db.query(ModeleDocument).filter(ModeleDocument.id == modele_id).first()
    if not m:
        raise ValueError("modele_not_found")

    placeholders = _build_render_context(db, data)

    def _apply(t: str | None) -> str | None:
        if not t:
            return t
        out = str(t)
        for key, value in placeholders.items():
            out = out.replace("{{" + key + "}}", _escape_html(str(value)))
        return out

    return {"objet": _apply(m.objet), "contenu": _apply(m.contenu)}
