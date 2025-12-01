from datetime import datetime
from typing import Iterable
import logging

from sqlalchemy.orm import Session
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.orm.exc import ObjectDeletedError
from sqlalchemy import and_, or_, func, text

from src.models.evenement import Evenement
from src.models.type_evenement import TypeEvenement
from src.models.statut_evenement import StatutEvenement
from src.models.evenement_statut import EvenementStatut
from src.models.evenement_intervenant import EvenementIntervenant
from src.models.evenement_lien import EvenementLien
from src.models.evenement_envoi import EvenementEnvoi
from src.models.modele_document import ModeleDocument

from src.schemas.evenement import EvenementCreateSchema, EvenementUpdateSchema, TacheCreateSchema
from src.schemas.evenement_statut import EvenementStatutCreateSchema
from src.schemas.evenement_intervenant import EvenementIntervenantCreateSchema
from src.schemas.evenement_lien import EvenementLienCreateSchema
from src.schemas.evenement_envoi import EvenementEnvoiCreateSchema

logger = logging.getLogger("uvicorn.error")

# -------------------- Evenements --------------------
def list_evenements(
    db: Session,
    *,
    type_id: int | None = None,
    statut: str | None = None,
    client_id: int | None = None,
    affaire_id: int | None = None,
    support_id: int | None = None,
    intervenant: str | None = None,
    categorie: str | None = None,
) -> Iterable[Evenement]:
    q = db.query(Evenement)
    if type_id is not None:
        q = q.filter(Evenement.type_id == type_id)
    if statut is not None:
        q = q.filter(Evenement.statut == statut)
    if client_id is not None:
        q = q.filter(Evenement.client_id == client_id)
    if affaire_id is not None:
        q = q.filter(Evenement.affaire_id == affaire_id)
    if support_id is not None:
        q = q.filter(Evenement.support_id == support_id)
    if categorie is not None:
        q = q.join(TypeEvenement, TypeEvenement.id == Evenement.type_id).filter(TypeEvenement.categorie == categorie)
    if intervenant is not None:
        q = q.join(EvenementIntervenant, EvenementIntervenant.evenement_id == Evenement.id).filter(
            or_(
                func.lower(EvenementIntervenant.nom_intervenant) == func.lower(intervenant),
                func.lower(EvenementIntervenant.role) == func.lower(intervenant),
            )
        )
    q = q.order_by(Evenement.date_evenement.desc())
    return q.all()


def get_evenement(db: Session, evenement_id: int) -> Evenement | None:
    return db.query(Evenement).filter(Evenement.id == evenement_id).first()


def create_evenement(db: Session, payload: EvenementCreateSchema) -> Evenement:
    from datetime import datetime as _dt
    started_at = _dt.utcnow()
    ev = Evenement(
        type_id=payload.type_id,
        client_id=payload.client_id,
        affaire_id=payload.affaire_id,
        support_id=payload.support_id,
        date_evenement=payload.date_evenement,
        statut=payload.statut or "à faire",
        commentaire=payload.commentaire,
        utilisateur_responsable=payload.utilisateur_responsable,
        rh_id=payload.rh_id,
        statut_reclamation_id=payload.statut_reclamation_id,
    )
    try:
        logger.info("create_evenement payload=%s", payload.dict() if hasattr(payload, "dict") else payload)
        db.add(ev)
        db.flush()  # assign PK before commit
        ev_id = ev.id
        db.commit()
        logger.info("create_evenement success ev_id=%s", ev_id)
    except Exception as exc:
        db.rollback()
        logger.exception("create_evenement failed", exc_info=True)
        raise
    try:
        if ev_id in (None, 0):
            logger.warning("create_evenement returned id=%s, attempting lookup", ev_id)
            row = (
                db.query(Evenement)
                .filter(Evenement.client_id == payload.client_id)
                .filter(Evenement.affaire_id == payload.affaire_id)
                .filter(Evenement.type_id == payload.type_id)
                .filter(Evenement.date_evenement >= started_at.replace(microsecond=0))
                .order_by(Evenement.id.desc())
                .first()
            )
            if row:
                ev_id = row.id
                return row
        return db.query(Evenement).filter(Evenement.id == ev_id).first()
    except Exception:
        return ev


def update_evenement(db: Session, evenement_id: int, payload: EvenementUpdateSchema) -> Evenement | None:
    ev = get_evenement(db, evenement_id)
    if not ev:
        return None
    for k, v in payload.dict(exclude_unset=True).items():
        setattr(ev, k, v)
    db.commit()
    db.refresh(ev)
    return ev


def delete_evenement(db: Session, evenement_id: int) -> bool:
    ev = get_evenement(db, evenement_id)
    if not ev:
        return False
    db.delete(ev)
    db.commit()
    return True


# -------------------- Statuts --------------------
def list_statuts(db: Session) -> Iterable[StatutEvenement]:
    return db.query(StatutEvenement).order_by(StatutEvenement.id.asc()).all()


def create_statut(db: Session, libelle: str) -> StatutEvenement:
    st = StatutEvenement(libelle=libelle)
    db.add(st)
    db.commit()
    db.refresh(st)
    return st


def list_types(db: Session) -> Iterable[TypeEvenement]:
    return db.query(TypeEvenement).order_by(TypeEvenement.id.asc()).all()


def create_type(db: Session, libelle: str, categorie: str) -> TypeEvenement:
    te = TypeEvenement(libelle=libelle, categorie=categorie)
    db.add(te)
    db.commit()
    db.refresh(te)
    return te


def ensure_type(db: Session, libelle: str, categorie: str | None = None) -> TypeEvenement:
    q = db.query(TypeEvenement).filter(func.lower(TypeEvenement.libelle) == func.lower(libelle)).first()
    if q:
        return q
    return create_type(db, libelle, categorie or "tache")


def create_tache(db: Session, payload: TacheCreateSchema) -> Evenement:
    from datetime import datetime as _dt
    started_at = _dt.utcnow()
    t = ensure_type(db, payload.type_libelle, payload.categorie or "tache")
    ev = Evenement(
        type_id=t.id,
        client_id=payload.client_id,
        affaire_id=payload.affaire_id,
        support_id=payload.support_id,
        date_evenement=payload.date_evenement or datetime.utcnow(),
        statut=payload.statut or "à faire",
        commentaire=payload.commentaire,
        utilisateur_responsable=payload.utilisateur_responsable,
        rh_id=payload.rh_id,
        statut_reclamation_id=payload.statut_reclamation_id,
    )
    try:
        logger.info("create_tache payload=%s", payload.dict() if hasattr(payload, "dict") else payload)
        db.add(ev)
        db.flush()
        ev_id = ev.id
        db.commit()
        logger.info("create_tache success ev_id=%s", ev_id)
    except Exception as exc:
        db.rollback()
        logger.exception("create_tache failed", exc_info=True)
        raise
    try:
        if ev_id in (None, 0):
            logger.warning("create_tache returned id=%s, attempting lookup", ev_id)
            row = (
                db.query(Evenement)
                .filter(Evenement.client_id == payload.client_id)
                .filter(Evenement.affaire_id == payload.affaire_id)
                .filter(Evenement.type_id == t.id)
                .filter(Evenement.date_evenement >= started_at.replace(microsecond=0))
                .order_by(Evenement.id.desc())
                .first()
            )
            if row:
                ev_id = row.id
                return row
        return db.query(Evenement).filter(Evenement.id == ev_id).first()
    except Exception:
        return ev


def add_statut_to_evenement(db: Session, evenement_id: int, payload: EvenementStatutCreateSchema) -> EvenementStatut | str:
    ev = get_evenement(db, evenement_id)
    if not ev:
        return "evenement_not_found"

    # Interdire la clôture si dépendances bloquantes
    statut_obj = db.query(StatutEvenement).filter(StatutEvenement.id == payload.statut_id).first()
    if not statut_obj:
        return "statut_not_found"
    new_label = statut_obj.libelle

    if new_label.lower() in ("terminé", "termine", "cloturé", "cloture", "clôturé", "annulé", "annule"):
        # vérifier s'il existe un lien de type 'bloque' où source n'est pas terminé et cible = evenement_id
        blocking = (
            db.query(Evenement)
            .join(EvenementLien, EvenementLien.evenement_source_id == Evenement.id)
            .filter(
                EvenementLien.type_lien == "bloque",
                EvenementLien.evenement_cible_id == evenement_id,
                Evenement.statut != "terminé",
            )
            .first()
        )
        if blocking:
            return "dependency_blocking"

    # Empêcher une régression de statut (redescendre de niveau)
    def _norm_label(x: str | None) -> str | None:
        if not x:
            return None
        s = x.strip().lower()
        # normalisations simples sans lib externe
        s = s.replace("à", "a").replace("é", "e").replace("è", "e").replace("ê", "e").replace("û", "u").replace("ô", "o")
        s = s.replace("ï", "i").replace("î", "i").replace("ç", "c")
        return s

    order = ["a faire", "en cours", "en attente", "termine", "annule"]
    cur_label = _norm_label(getattr(ev, "statut", None))
    new_label_norm = _norm_label(new_label)
    try:
        cur_pos = order.index(cur_label) if cur_label in order else -1
    except ValueError:
        cur_pos = -1
    try:
        new_pos = order.index(new_label_norm) if new_label_norm in order else -1
    except ValueError:
        new_pos = -1
    if cur_pos >= 0 and new_pos >= 0 and new_pos < cur_pos:
        return "status_downgrade_forbidden"

    es = EvenementStatut(
        evenement_id=evenement_id,
        statut_id=payload.statut_id,
        date_statut=payload.date_statut or datetime.utcnow(),
        utilisateur_responsable=payload.utilisateur_responsable,
        commentaire=payload.commentaire,
    )
    db.add(es)

    # Mettre à jour le statut courant de l'évènement
    ev.statut = new_label
    db.add(ev)
    db.commit()
    db.refresh(es)

    # Déclenchements éventuels (best‑effort)
    if new_label.lower() in ("terminé", "termine", "cloturé", "cloture", "clôturé"):
        _maybe_create_followups(db, evenement_id)

    return es


def get_evenement_statuts(db: Session, evenement_id: int) -> Iterable[EvenementStatut]:
    return (
        db.query(EvenementStatut)
        .filter(EvenementStatut.evenement_id == evenement_id)
        .order_by(EvenementStatut.date_statut.asc())
        .all()
    )


# -------------------- Intervenants --------------------
def add_intervenant(db: Session, evenement_id: int, payload: EvenementIntervenantCreateSchema) -> EvenementIntervenant | str:
    if not get_evenement(db, evenement_id):
        return "evenement_not_found"
    it = EvenementIntervenant(
        evenement_id=evenement_id,
        role=payload.role,
        nom_intervenant=payload.nom_intervenant,
        contact=payload.contact,
    )
    db.add(it)
    db.commit()
    db.refresh(it)
    return it


def list_intervenants(db: Session, evenement_id: int) -> Iterable[EvenementIntervenant]:
    return (
        db.query(EvenementIntervenant)
        .filter(EvenementIntervenant.evenement_id == evenement_id)
        .order_by(EvenementIntervenant.id.asc())
        .all()
    )


def delete_intervenant(db: Session, intervenant_id: int) -> bool:
    it = db.query(EvenementIntervenant).filter(EvenementIntervenant.id == intervenant_id).first()
    if not it:
        return False
    db.delete(it)
    db.commit()
    return True


# -------------------- Liens --------------------
def add_lien(db: Session, evenement_id: int, payload: EvenementLienCreateSchema) -> EvenementLien | str:
    if not get_evenement(db, evenement_id):
        return "evenement_not_found"
    if not get_evenement(db, payload.evenement_cible_id):
        return "evenement_cible_not_found"
    ln = EvenementLien(
        evenement_source_id=evenement_id,
        evenement_cible_id=payload.evenement_cible_id,
        type_lien=payload.type_lien,
    )
    db.add(ln)
    db.commit()
    db.refresh(ln)
    return ln


def list_liens(db: Session, evenement_id: int) -> Iterable[EvenementLien]:
    return (
        db.query(EvenementLien)
        .filter(
            or_(
                EvenementLien.evenement_source_id == evenement_id,
                EvenementLien.evenement_cible_id == evenement_id,
            )
        )
        .order_by(EvenementLien.id.asc())
        .all()
    )


def delete_lien(db: Session, lien_id: int) -> bool:
    ln = db.query(EvenementLien).filter(EvenementLien.id == lien_id).first()
    if not ln:
        return False
    db.delete(ln)
    db.commit()
    return True


# -------------------- Envois --------------------
def create_envoi(db: Session, evenement_id: int, payload: EvenementEnvoiCreateSchema) -> EvenementEnvoi | str:
    if not get_evenement(db, evenement_id):
        return "evenement_not_found"

    contenu = payload.contenu
    objet = payload.objet
    if payload.modele_id and (not contenu or not objet):
        # Rendu automatique depuis modèle si fourni
        tpl = db.query(ModeleDocument).filter(ModeleDocument.id == payload.modele_id).first()
        if not tpl:
            return "modele_not_found"
        rendered = render_modele_from_obj(tpl, payload.placeholders or {})
        contenu = contenu or rendered.get("contenu")
        objet = objet or rendered.get("objet")

    ev = EvenementEnvoi(
        evenement_id=evenement_id,
        canal=payload.canal,
        destinataire=payload.destinataire,
        objet=objet,
        contenu=contenu,
        date_envoi=payload.date_envoi or datetime.utcnow(),
        statut=payload.statut or "préparé",
        modele_id=payload.modele_id,
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev


def update_envoi_statut(db: Session, envoi_id: int, statut: str) -> EvenementEnvoi | None:
    ev = db.query(EvenementEnvoi).filter(EvenementEnvoi.id == envoi_id).first()
    if not ev:
        return None
    ev.statut = statut
    db.commit()
    db.refresh(ev)
    return ev


def list_envois(db: Session, evenement_id: int) -> Iterable[EvenementEnvoi]:
    return (
        db.query(EvenementEnvoi)
        .filter(EvenementEnvoi.evenement_id == evenement_id)
        .order_by(EvenementEnvoi.date_envoi.desc())
        .all()
    )


# -------------------- Modèles de documents --------------------
def list_modeles(db: Session) -> Iterable[ModeleDocument]:
    return db.query(ModeleDocument).order_by(ModeleDocument.id.asc()).all()


def get_modele(db: Session, modele_id: int) -> ModeleDocument | None:
    return db.query(ModeleDocument).filter(ModeleDocument.id == modele_id).first()


def create_modele(db: Session, nom: str, canal: str, contenu: str, objet: str | None = None, actif: int | None = 1) -> ModeleDocument:
    m = ModeleDocument(nom=nom, canal=canal, contenu=contenu, objet=objet, actif=actif)
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def update_modele(db: Session, modele_id: int, **fields) -> ModeleDocument | None:
    m = get_modele(db, modele_id)
    if not m:
        return None
    for k, v in fields.items():
        if v is not None:
            setattr(m, k, v)
    db.commit()
    db.refresh(m)
    return m


def delete_modele(db: Session, modele_id: int) -> bool:
    m = get_modele(db, modele_id)
    if not m:
        return False
    db.delete(m)
    db.commit()
    return True


# -------------------- Reporting (vues) --------------------
def vue_reclamations(db: Session, *, statut: str | None = None, client_id: int | None = None):
    conds = []
    params = {}
    if statut:
        conds.append("statut = :statut")
        params["statut"] = statut
    if client_id is not None:
        conds.append("client_id = :cid")
        params["cid"] = client_id
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    sql = f"SELECT * FROM vue_reclamations {where} ORDER BY date_evenement DESC"
    return db.execute(text(sql), params).fetchall()


def vue_suivi_evenement(db: Session, *, statut: str | None = None, type_id: int | None = None):
    conds = []
    params = {}
    if statut:
        conds.append("statut = :statut")
        params["statut"] = statut
    if type_id is not None:
        conds.append("type_evenement IN (SELECT libelle FROM mariadb_type_evenement WHERE id = :tid)")
        params["tid"] = type_id
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    sql = f"SELECT * FROM vue_suivi_evenement {where} ORDER BY date_evenement DESC"
    return db.execute(text(sql), params).fetchall()


# -------------------- Helpers --------------------
def render_modele_from_obj(m: ModeleDocument, placeholders: dict[str, str]) -> dict[str, str]:
    def _apply(t: str | None):
        if not t:
            return t
        out = t
        for k, v in placeholders.items():
            out = out.replace("{{" + k + "}}", str(v))
        return out

    return {
        "objet": _apply(m.objet),
        "contenu": _apply(m.contenu),
    }


def _maybe_create_followups(db: Session, evenement_id: int) -> None:
    # Placeholder: si des règles spécifiques de création existent, les brancher ici.
    # À ce stade, nous n'avons pas d'information suffisante pour créer un "evenement cible" absent.
    return None
