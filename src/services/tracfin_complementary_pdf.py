from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfgen import canvas


PAGE_W, PAGE_H = A4  # 595.2755905511812 x 841.8897637795277


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def default_tracfin_background_path() -> Path:
    # Kept for backward-compatibility with earlier iterations (no longer required).
    return _root_dir() / "documents" / "templates" / "tracfin.png"


def _s(v: Any) -> str:
    return "" if v is None else str(v)


def _fmt_eur(v: Any) -> str:
    try:
        f = float(v)
    except Exception:
        return ""
    try:
        return "{:,.0f} €".format(f).replace(",", " ")
    except Exception:
        return f"{f} €"


def _fmt_pct(v: Any) -> str:
    try:
        f = float(v)
    except Exception:
        return ""
    try:
        return "{:,.1f} %".format(f).replace(",", " ").replace(".", ",")
    except Exception:
        return f"{f} %"


@dataclass(frozen=True)
class _Ctx:
    c: canvas.Canvas
    x: float
    y: float
    w: float
    margin_bottom: float = 36

    def ensure(self, h: float) -> "_Ctx":
        if self.y - h < self.margin_bottom:
            self.c.showPage()
            return _Ctx(self.c, self.x, PAGE_H - 36, self.w, self.margin_bottom)
        return self

    def down(self, dy: float) -> "_Ctx":
        return _Ctx(self.c, self.x, self.y - dy, self.w, self.margin_bottom)


GREY = colors.Color(0.90, 0.90, 0.90)
BLACK = colors.black


def _draw_box(c: canvas.Canvas, x: float, y: float, w: float, h: float, *, stroke: int = 1, fill: int = 0, fill_color=None, lw: float = 1):
    c.setLineWidth(lw)
    if fill_color is not None:
        c.setFillColor(fill_color)
    c.rect(x, y, w, h, stroke=stroke, fill=fill)


def _draw_text(c: canvas.Canvas, x: float, y: float, s: str, *, size: float = 9, bold: bool = False, color=BLACK):
    c.setFillColor(color)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    c.drawString(x, y, s)


def _draw_centered(c: canvas.Canvas, x: float, y: float, w: float, s: str, *, size: float = 11, bold: bool = True):
    c.setFillColor(BLACK)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    c.drawCentredString(x + w / 2, y, s)


def _p_style(size: float = 9, leading: float | None = None, bold: bool = False) -> ParagraphStyle:
    return ParagraphStyle(
        name="p",
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size,
        leading=float(leading if leading is not None else size * 1.25),
        textColor=BLACK,
    )


def _measure_paragraph(text: str, w: float, *, size: float = 9, leading: float | None = None, bold: bool = False) -> float:
    p = Paragraph((_s(text) or "").replace("\n", "<br/>"), _p_style(size=size, leading=leading, bold=bold))
    _, h = p.wrap(w, PAGE_H)
    return float(h)


def _draw_paragraph(c: canvas.Canvas, x: float, y_top: float, w: float, text: str, *, size: float = 9, leading: float | None = None, bold: bool = False) -> float:
    p = Paragraph((_s(text) or "").replace("\n", "<br/>"), _p_style(size=size, leading=leading, bold=bold))
    _, h = p.wrap(w, PAGE_H)
    p.drawOn(c, x, y_top - h)
    return float(h)


def _draw_label_value(
    ctx: _Ctx,
    label: str,
    value: str,
    *,
    value_w: float,
    h: float = 14,
    font_size: float = 9,
    label_w: float | None = None,
) -> _Ctx:
    ctx = ctx.ensure(h + 4)
    fixed_label_w = float(label_w) if label_w is not None else max(120, ctx.w - value_w - 8)
    _draw_text(ctx.c, ctx.x, ctx.y - 10, label, size=font_size, bold=False)
    _draw_box(ctx.c, ctx.x + fixed_label_w + 8, ctx.y - h - 2, value_w, h, stroke=0, fill=1, fill_color=GREY, lw=0)
    _draw_text(ctx.c, ctx.x + fixed_label_w + 12, ctx.y - 10, value, size=font_size, bold=False)
    return ctx.down(h + 6)


def _draw_underlined_title(ctx: _Ctx, title: str) -> _Ctx:
    ctx = ctx.ensure(18)
    _draw_text(ctx.c, ctx.x, ctx.y - 12, title, size=10, bold=True)
    ctx.c.setLineWidth(1)
    ctx.c.line(ctx.x, ctx.y - 14, ctx.x + min(160, ctx.w), ctx.y - 14)
    return ctx.down(18)


def _draw_section_header(ctx: _Ctx, title: str) -> _Ctx:
    ctx = ctx.ensure(22)
    _draw_box(ctx.c, ctx.x, ctx.y - 18, ctx.w, 18, stroke=1, fill=0, lw=1)
    _draw_centered(ctx.c, ctx.x, ctx.y - 14, ctx.w, title, size=10, bold=True)
    return ctx.down(24)


def _draw_choice_pills(
    ctx: _Ctx,
    title: str,
    items: list[tuple[str, bool]],
    *,
    cols: int,
    row_h: float = 18,
    gap: float = 8,
    font_size: float = 8.5,
) -> _Ctx:
    has_title = bool(title and title.strip())
    header_h = 18 if has_title else 0
    ctx = ctx.ensure(header_h + (row_h + gap) * ((len(items) + cols - 1) // cols) + 6)
    if has_title:
        _draw_text(ctx.c, ctx.x, ctx.y - 12, title, size=9, bold=True)
        y = ctx.y - 18
    else:
        y = ctx.y - 2
    w_cell = (ctx.w - (cols - 1) * gap) / cols
    for idx, (label, active) in enumerate(items):
        r = idx // cols
        col = idx % cols
        x = ctx.x + col * (w_cell + gap)
        y_cell = y - r * (row_h + gap) - row_h
        _draw_box(ctx.c, x, y_cell, w_cell, row_h, stroke=1, fill=1 if active else 0, fill_color=GREY if active else None, lw=1)
        ctx.c.setFont("Helvetica", font_size)
        ctx.c.setFillColor(BLACK)
        ctx.c.drawCentredString(x + w_cell / 2, y_cell + 5, label)
    return ctx.down(header_h + (row_h + gap) * ((len(items) + cols - 1) // cols) + 6)


def _draw_choice_list_box(
    ctx: _Ctx,
    title: str,
    items: list[tuple[str, bool]],
    *,
    box_w: float,
    box_h: float,
    font_size: float = 8,
) -> _Ctx:
    ctx = ctx.ensure(box_h + 18)
    _draw_box(ctx.c, ctx.x, ctx.y - box_h - 6, box_w, box_h, stroke=1, fill=0, lw=1)
    _draw_text(ctx.c, ctx.x + 8, ctx.y - 16, title, size=9, bold=True)
    y = ctx.y - 32
    line_h = 12
    for label, active in items:
        if y - line_h < ctx.y - box_h - 6 + 6:
            break
        if active:
            _draw_box(ctx.c, ctx.x + 6, y - 10, box_w - 12, 12, stroke=0, fill=1, fill_color=GREY, lw=0)
        _draw_text(ctx.c, ctx.x + 10, y - 8, label, size=font_size, bold=False)
        y -= line_h
    return ctx


def _build_patrimoine_summary(data: dict[str, Any]) -> str:
    total = _s(data.get("patrimoine_total_str")).strip()
    net = _s(data.get("patrimoine_net_str")).strip()
    parts = []
    if total:
        parts.append(f"Total actifs : {total}")
    if net:
        parts.append(f"Patrimoine net estimé : {net}")
    breakdown = data.get("patrimoine_breakdown") or []
    if isinstance(breakdown, list) and breakdown:
        # Keep it compact: top 4 categories.
        top = []
        for row in breakdown[:4]:
            try:
                lbl = _s(row.get("label")).strip()
                amt = _s(row.get("amount_str")).strip()
            except Exception:
                continue
            if lbl and amt:
                top.append(f"{lbl} ({amt})")
        if top:
            parts.append("Répartition : " + ", ".join(top))
    return " — ".join(parts)


def _draw_two_col_amount_row(
    ctx: _Ctx,
    left_label: str,
    left_value: str,
    right_label: str,
    right_value: str,
    *,
    label_w: float = 150,
    h: float = 14,
    font_size: float = 9,
    gap: float = 16,
) -> _Ctx:
    ctx = ctx.ensure(h + 8)
    col_w = (ctx.w - gap) / 2

    def _col(x: float, label: str, value: str) -> None:
        _draw_text(ctx.c, x, ctx.y - 10, label, size=font_size, bold=False)
        box_w = max(60, col_w - label_w - 8)
        _draw_box(ctx.c, x + label_w + 8, ctx.y - h - 2, box_w, h, stroke=0, fill=1, fill_color=GREY, lw=0)
        _draw_text(ctx.c, x + label_w + 12, ctx.y - 10, value, size=font_size, bold=False)

    _col(ctx.x, left_label, left_value)
    _col(ctx.x + col_w + gap, right_label, right_value)
    return ctx.down(h + 6)


def build_tracfin_complementary_pdf_bytes(data: dict[str, Any], *, background_path: Path | None = None) -> bytes:
    """
    Génère une fiche TRACFIN propre (sans dépendance à un fond PNG).
    Le paramètre background_path est conservé pour compatibilité mais ignoré.
    """
    _ = background_path

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    margin_x = 36
    ctx = _Ctx(c=c, x=margin_x, y=PAGE_H - 36, w=PAGE_W - 2 * margin_x)

    # --- Header ---
    ctx = ctx.ensure(70)
    _draw_box(c, ctx.x, ctx.y - 18, ctx.w, 18, stroke=1, fill=0, lw=1)
    _draw_centered(c, ctx.x, ctx.y - 14, ctx.w, "Fiche de renseignement complémentaire", size=11, bold=True)
    _draw_centered(c, ctx.x, ctx.y - 34, ctx.w, "Personnes physiques résidentes", size=11, bold=True)
    ctx = ctx.down(54)

    # --- Interlocuteur / Société ---
    ctx = _draw_label_value(ctx, "Interlocuteur commercial :", _s(data.get("interlocuteur_commercial")).strip(), value_w=360)
    ctx = _draw_label_value(ctx, "Société (courtier) :", _s(data.get("societe")).strip(), value_w=360)

    # --- Client block ---
    # Keep value boxes aligned left for these 3 lines (same label column width)
    header_label_w = 160
    ctx = _draw_label_value(ctx, "Nom du client :", _s(data.get("client_nom")).strip(), value_w=360, label_w=header_label_w)
    ctx = _draw_label_value(ctx, "Prénom du client :", _s(data.get("client_prenom")).strip(), value_w=360, label_w=header_label_w)
    ctx = _draw_label_value(ctx, "Date de naissance :", _s(data.get("date_naissance")).strip(), value_w=360, label_w=header_label_w)
    lieu = " ".join([p for p in [_s(data.get("lieu_naissance_cp")).strip(), _s(data.get("lieu_naissance_ville")).strip()] if p]).strip()
    pays_naissance = _s(data.get("lieu_naissance_pays")).strip()
    lieu_full = ", ".join([p for p in [lieu, pays_naissance] if p]).strip()
    ctx = _draw_label_value(ctx, "Lieu de naissance (CP, ville, pays) :", lieu_full, value_w=360)

    # --- Contrat ---
    ctx = _draw_label_value(ctx, "Nom du contrat :", _s(data.get("nom_contrat")).strip(), value_w=360)
    ctx = ctx.down(6)

    # --- Données personnelles ---
    ctx = _draw_section_header(ctx, "Données personnelles et patrimoniales")

    # Adresse + courriel
    adresse_lines = []
    adr1 = " ".join([p for p in [_s(data.get("adresse_numero")).strip(), _s(data.get("adresse_rue")).strip()] if p]).strip()
    adr2 = " ".join([p for p in [_s(data.get("adresse_code_postal")).strip(), _s(data.get("adresse_ville")).strip()] if p]).strip()
    adr3 = _s(data.get("adresse_pays")).strip()
    if adr1:
        adresse_lines.append(adr1)
    if adr2:
        adresse_lines.append(adr2)
    if adr3:
        adresse_lines.append(adr3)
    adresse_txt = "\n".join(adresse_lines)

    ctx = ctx.ensure(64)
    _draw_text(c, ctx.x, ctx.y - 12, "Adresse du client :", size=9, bold=False)
    _draw_box(c, ctx.x + 130, ctx.y - 52, ctx.w - 130, 44, stroke=0, fill=1, fill_color=GREY, lw=0)
    if adresse_txt:
        _draw_paragraph(c, ctx.x + 136, ctx.y - 12, ctx.w - 142, adresse_txt, size=9)
    ctx = ctx.down(52)

    ctx = _draw_label_value(ctx, "Courriel :", _s(data.get("courriel")).strip(), value_w=360)
    ctx = ctx.down(4)

    # Activité
    ctx = _draw_underlined_title(ctx, "Activité")
    ctx = _draw_label_value(ctx, "Profession :", _s(data.get("profession_precise")).strip(), value_w=360)
    ctx = ctx.down(6)

    # 3 blocs (secteur / revenus / patrimoine)
    gap = 14
    col_w = (ctx.w - 2 * gap) / 3
    box_h = 160
    ctx = ctx.ensure(box_h + 10)
    x0 = ctx.x
    y0 = ctx.y

    secteur_selected = _s(data.get("secteur_activite")).strip()
    rev_selected = _s(data.get("tranche_revenus")).strip()
    patr_selected = _s(data.get("tranche_patrimoine")).strip()

    secteur_items = [
        ("Agriculteur exploitant", secteur_selected == "agriculteur"),
        ("Artisan", secteur_selected == "artisan"),
        ("Artiste", secteur_selected == "artiste"),
        ("Cadre", secteur_selected == "cadre"),
        ("Commerçant", secteur_selected == "commercant"),
        ("Chef d'entreprise", secteur_selected == "chef_entreprise"),
        ("Ouvrier", secteur_selected == "ouvrier"),
        ("Employé", secteur_selected == "employe"),
        ("Retraité", secteur_selected == "retraite"),
        ("Profession intermédiaire", secteur_selected == "profession_intermediaire"),
        ("Profession libérale", secteur_selected == "profession_liberale"),
        ("Professeur", secteur_selected == "professeur"),
        ("Scientifique", secteur_selected == "scientifique"),
        ("Sans activité", secteur_selected == "sans_activite"),
    ]
    rev_items = [
        ("0 à 50 000 €", rev_selected == "rev_0_50k"),
        ("50 à 80 000 €", rev_selected == "rev_50_80k"),
        ("80 à 120 000 €", rev_selected == "rev_80_120k"),
        ("120 à 160 000 €", rev_selected == "rev_120_160k"),
        ("160 à 300 000 €", rev_selected == "rev_160_300k"),
        ("> 300 000 €", rev_selected == "rev_300k_plus"),
    ]
    patr_items = [
        ("< 150 k", patr_selected == "patr_lt_150k"),
        ("150 à 750 k", patr_selected == "patr_150_750k"),
        ("750 k à 1,5 M", patr_selected == "patr_750_1_5m"),
        ("1,5 à 5 M", patr_selected == "patr_1_5_5m"),
        ("5 à 15 M", patr_selected == "patr_5_15m"),
        ("> 15 M", patr_selected == "patr_15m_plus"),
    ]

    _draw_choice_list_box(_Ctx(c, x0, y0, col_w), "Secteur d'activité", secteur_items, box_w=col_w, box_h=box_h)
    _draw_choice_list_box(_Ctx(c, x0 + col_w + gap, y0, col_w), "Tranche de revenu", rev_items, box_w=col_w, box_h=box_h)
    _draw_choice_list_box(_Ctx(c, x0 + 2 * (col_w + gap), y0, col_w), "Tranche de patrimoine", patr_items, box_w=col_w, box_h=box_h)
    ctx = ctx.down(box_h + 14)

    # PPE
    ctx = _draw_section_header(ctx, "PPE (Personne Politiquement Exposée)")
    q1 = (
        "Exercez-vous ou avez-vous cessé d'exercer depuis moins d'un an des fonctions politiques, "
        "juridictionnelles ou administratives (exercice d'un mandat électif, social, ..) ?"
    )
    ctx = ctx.ensure(44)
    hq1 = _draw_paragraph(c, ctx.x, ctx.y - 2, ctx.w, q1, size=8.2, leading=10)
    ctx = ctx.down(hq1 + 2)

    ppe_self = _s(data.get("ppe_self")).strip().lower()
    pills = [("oui", ppe_self == "oui"), ("non", ppe_self == "non")]
    ctx = _draw_choice_pills(ctx, "", pills, cols=2, row_h=16, gap=10, font_size=9)

    q2 = (
        "Un des membres directs de votre famille, ou une des personnes de votre entourage étroitement associées, "
        "exercent-ils ou ont-ils cessé d'exercer depuis moins d'un an une fonction publique, juridictionnelle ou administrative ?"
    )
    ctx = ctx.ensure(48)
    hq2 = _draw_paragraph(c, ctx.x, ctx.y - 2, ctx.w, q2, size=8.2, leading=10)
    ctx = ctx.down(hq2 + 2)

    ppe_family = _s(data.get("ppe_family")).strip().lower()
    pills2 = [("oui", ppe_family == "oui"), ("non", ppe_family == "non")]
    ctx = _draw_choice_pills(ctx, "", pills2, cols=2, row_h=16, gap=10, font_size=9)

    if ppe_family == "oui":
        details = _s(data.get("ppe_family_details")).strip()
        if details:
            ctx = ctx.ensure(48)
            _draw_text(c, ctx.x, ctx.y - 12, "Si oui, liste des proches :", size=8.5, bold=True)
            _draw_box(c, ctx.x, ctx.y - 44, ctx.w, 30, stroke=0, fill=1, fill_color=GREY, lw=0)
            _draw_paragraph(c, ctx.x + 6, ctx.y - 16, ctx.w - 12, details, size=9)
            ctx = ctx.down(44)

    # ---- Page 2: Estimation patrimoine + Opération + Vigilance + Justificatifs + signatures ----
    c.showPage()
    ctx = _Ctx(c=c, x=margin_x, y=PAGE_H - 36, w=PAGE_W - 2 * margin_x)

    # Estimation du patrimoine
    ctx = _draw_section_header(ctx, "Estimation du patrimoine")
    # Table layout (2 columns)
    ctx = _draw_two_col_amount_row(
        ctx,
        "Total des actifs :",
        _s(data.get("patrimoine_total_str")).strip(),
        "Patrimoine net estimé :",
        _s(data.get("patrimoine_net_str")).strip(),
    )
    ctx = _draw_two_col_amount_row(
        ctx,
        "Biens professionnels :",
        _s(data.get("patrimoine_biens_professionnels_str")).strip(),
        "Contrats d'assurances :",
        _s(data.get("patrimoine_contrats_assurances_str")).strip(),
    )
    ctx = _draw_two_col_amount_row(
        ctx,
        "Immobilier :",
        _s(data.get("patrimoine_immobilier_str")).strip(),
        "Comptes :",
        _s(data.get("patrimoine_comptes_str")).strip(),
    )
    ctx = _draw_two_col_amount_row(
        ctx,
        "Valeurs mobilières :",
        _s(data.get("patrimoine_valeurs_mobilieres_str")).strip(),
        "Autres :",
        _s(data.get("patrimoine_autres_str")).strip(),
    )

    ctx = _draw_section_header(ctx, "Opération")

    invest_total = _s(data.get("invest_total_str")).strip()
    invest_pct = _s(data.get("invest_pct_str")).strip()
    ctx = _draw_label_value(ctx, "Montant total à investir :", invest_total, value_w=200)
    ctx = _draw_label_value(ctx, "% de l'investissement vs patrimoine :", invest_pct, value_w=200)
    ctx = ctx.down(6)

    # Objet de l'opération
    ctx = _draw_underlined_title(ctx, "Objet de l'opération")
    sel_ops: set[str] = set()
    raw_sel = data.get("operation_objets_selected") or []
    if isinstance(raw_sel, (list, tuple, set)):
        for it in raw_sel:
            s = _s(it).strip().lower()
            if s:
                sel_ops.add(s)
    else:
        s = _s(raw_sel).strip().lower()
        if s:
            sel_ops.add(s)

    ops = [
        ("Souscription", "souscription"),
        ("Versement complémentaire", "versement"),
        ("Rachat", "rachat"),
        ("Avances", "avances"),
        ("Renonciation", "renonciation"),
        ("Remboursement d'avance", "remboursement"),
        ("Mise en garantie", "garantie"),
        ("Autres", "autre"),
    ]
    ops_items = [(label, key in sel_ops) for (label, key) in ops]
    ctx = _draw_choice_pills(ctx, "", ops_items, cols=4, row_h=18, gap=8, font_size=8.2)
    ctx = ctx.down(8)

    # Raison de la vigilance
    ctx = _draw_underlined_title(ctx, "Raison de la vigilance")
    reasons = data.get("raison_vigilance_labels") or []
    if isinstance(reasons, (list, tuple)) and reasons:
        text = "\n".join([f"• {_s(x).strip()}" for x in reasons if _s(x).strip()])
        h = _measure_paragraph(text, ctx.w, size=9, leading=11)
        ctx = ctx.ensure(h + 12)
        _draw_box(c, ctx.x, ctx.y - (h + 8), ctx.w, h + 8, stroke=0, fill=1, fill_color=GREY, lw=0)
        _draw_paragraph(c, ctx.x + 6, ctx.y - 4, ctx.w - 12, text, size=9, leading=11)
        ctx = ctx.down(h + 14)
    else:
        ctx = _draw_label_value(ctx, "—", "", value_w=360)

    # Justificatifs & précisions
    def block(title: str, value: str, *, min_h: float) -> None:
        nonlocal ctx
        value = _s(value).strip()
        ctx = ctx.ensure(18 + min_h + 8)
        _draw_text(c, ctx.x, ctx.y - 12, title, size=9, bold=True)
        if not value:
            _draw_box(c, ctx.x, ctx.y - (min_h + 22), ctx.w, min_h, stroke=0, fill=1, fill_color=GREY, lw=0)
            ctx = ctx.down(min_h + 28)
            return
        htxt = _measure_paragraph(value, ctx.w - 12, size=9, leading=11)
        h_box = max(min_h, htxt + 8)
        ctx = ctx.ensure(h_box + 26)
        _draw_box(c, ctx.x, ctx.y - (h_box + 22), ctx.w, h_box, stroke=0, fill=1, fill_color=GREY, lw=0)
        _draw_paragraph(c, ctx.x + 6, ctx.y - 18, ctx.w - 12, value, size=9, leading=11)
        ctx = ctx.down(h_box + 28)

    block("Précisions sur l'opération à l'origine des fonds", data.get("just_fonds"), min_h=44)
    block("Destination des fonds", data.get("just_destination"), min_h=34)
    block("Précisions et explications sur la finalité de la souscription ou de l'opération demandée", data.get("just_finalite"), min_h=34)
    block("Justificatifs produits lorsque requis", data.get("just_produits"), min_h=34)

    # Signatures
    ctx = ctx.ensure(90)
    fait_a = _s(data.get("fait_a")).strip()
    date_txt = _s(data.get("date_signature")).strip()
    # Keep values close to their labels (compact row)
    ctx = ctx.ensure(34)
    y_row = ctx.y - 14
    _draw_text(c, ctx.x, y_row, "Fait à :", size=9, bold=False)
    _draw_box(c, ctx.x + 40, y_row - 10, 170, 14, stroke=0, fill=1, fill_color=GREY, lw=0)
    _draw_text(c, ctx.x + 46, y_row, fait_a, size=9)

    _draw_text(c, ctx.x + 230, y_row, "Date :", size=9, bold=False)
    _draw_box(c, ctx.x + 268, y_row - 10, 120, 14, stroke=0, fill=1, fill_color=GREY, lw=0)
    _draw_text(c, ctx.x + 274, y_row, date_txt, size=9)
    ctx = ctx.down(34)

    sig_line_w = 240
    x_client = ctx.x
    x_courtier = ctx.x + ctx.w / 2 + 10
    _draw_text(c, x_client, ctx.y - 12, "Signature client", size=9, bold=True)
    c.setLineWidth(1)
    c.line(x_client, ctx.y - 30, x_client + sig_line_w, ctx.y - 30)
    _draw_text(c, x_courtier + 10, ctx.y - 12, "Signature courtier", size=9, bold=True)
    c.line(x_courtier, ctx.y - 30, x_courtier + sig_line_w, ctx.y - 30)

    c.save()
    return buffer.getvalue()
