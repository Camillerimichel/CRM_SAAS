from __future__ import annotations

import io
from functools import lru_cache
from typing import Any

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


PAGE_W, PAGE_H = A4  # 595.2755905511812 x 841.8897637795277


def _draw_text(
    c: canvas.Canvas,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 9,
    bold: bool = False,
    color=colors.black,
):
    c.setFillColor(color)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    c.drawString(x, y, text)


def _draw_paragraph(
    c: canvas.Canvas,
    x: float,
    y: float,
    text: str,
    *,
    w: float,
    size: float,
    leading: float,
) -> float:
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle

    style = ParagraphStyle(
        name="p",
        fontName="Helvetica",
        fontSize=size,
        leading=leading,
        textColor=colors.black,
    )
    p = Paragraph(text.replace("\n", "<br/>"), style)
    _, h = p.wrap(w, PAGE_H)
    p.drawOn(c, x, y - h)
    return float(h)


def _measure_paragraph_height(text: str, w: float, size: float, leading: float) -> float:
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle

    style = ParagraphStyle(
        name="m",
        fontName="Helvetica",
        fontSize=size,
        leading=leading,
        textColor=colors.black,
    )
    p = Paragraph(text.replace("\n", "<br/>"), style)
    _, h = p.wrap(w, PAGE_H)
    return float(h)


def _coerce_field_values(data: dict[str, Any]) -> dict[str, str]:
    coerced: dict[str, str] = {}
    for k, v in data.items():
        if isinstance(v, bool):
            coerced[k] = "/Yes" if v else "/Off"
        else:
            coerced[k] = "" if v is None else str(v)
    return coerced


def _build_template_pdf_bytes() -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    form = c.acroForm

    # Header
    _draw_text(c, 140, PAGE_H - 55, "Autocertification fiscale obligatoire FATCA et EIA", size=14, bold=True)
    c.setStrokeColor(colors.black)
    c.setLineWidth(1)
    c.rect(30, PAGE_H - 78, PAGE_W - 60, 18, stroke=1, fill=0)

    # Top-left: interlocuteur commercial
    _draw_text(c, 30, PAGE_H - 110, "Interlocuteur commercial", size=9, bold=False, color=colors.red)
    form.textfield(
        name="interlocuteur_commercial",
        x=170,
        y=PAGE_H - 124,
        width=200,
        height=14,
        fillColor=colors.lightgrey,
        borderWidth=0,
        textColor=colors.black,
        fontSize=9,
        maxlen=80,
    )

    # Top-right: produit/compagnie/date
    right_x = 310
    _draw_text(c, right_x, PAGE_H - 108, "Nom du produit", size=8)
    form.textfield(
        name="produit_nom",
        x=right_x + 120,
        y=PAGE_H - 124,
        width=PAGE_W - (right_x + 120) - 30,
        height=14,
        fillColor=colors.lightgrey,
        borderWidth=0,
        fontSize=8,
        maxlen=120,
    )
    _draw_text(c, right_x, PAGE_H - 124 - 18, "Compagnie d'assurance", size=8)
    form.textfield(
        name="compagnie_assurance",
        x=right_x + 120,
        y=PAGE_H - 124 - 18 - 14,
        width=PAGE_W - (right_x + 120) - 30,
        height=14,
        fillColor=colors.lightgrey,
        borderWidth=0,
        fontSize=8,
        maxlen=120,
    )
    _draw_text(c, right_x, PAGE_H - 124 - 36, "Date de l'opération", size=8)
    form.textfield(
        name="date_operation",
        x=right_x + 120,
        y=PAGE_H - 124 - 36 - 14,
        width=180,
        height=14,
        fillColor=colors.lightgrey,
        borderWidth=0,
        fontSize=8,
        maxlen=30,
    )

    # Opération (checkboxes)
    _draw_text(c, 30, PAGE_H - 150, "Opération", size=8, bold=True)
    form.checkbox(name="operation_souscription", x=120, y=PAGE_H - 162, size=10, borderWidth=1, fillColor=colors.white)
    _draw_text(c, 135, PAGE_H - 160, "Souscription", size=8)
    form.checkbox(name="operation_autre", x=220, y=PAGE_H - 162, size=10, borderWidth=1, fillColor=colors.white)
    _draw_text(c, 235, PAGE_H - 160, "Autre", size=8)

    def person_block(prefix: str, title: str, x0: float, y0: float):
        _draw_text(c, x0, y0, title, size=9, color=colors.red)
        y = y0 - 16
        label_x = x0
        field_x = x0 + 95
        field_w = 155
        row_h = 12

        _draw_text(c, label_x, y, "M. / Mme", size=8)
        form.textfield(name=f"{prefix}_civilite", x=field_x, y=y - 3, width=50, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=10)

        y -= 14
        _draw_text(c, label_x, y, "Nom", size=8)
        form.textfield(name=f"{prefix}_nom", x=field_x, y=y - 3, width=field_w, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=60)

        y -= 14
        _draw_text(c, label_x, y, "Nom de jeune fille", size=8)
        form.textfield(name=f"{prefix}_nom_jeune_fille", x=field_x, y=y - 3, width=field_w, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=60)

        y -= 14
        _draw_text(c, label_x, y, "Prénom d'usage", size=8)
        form.textfield(name=f"{prefix}_prenom_usage", x=field_x, y=y - 3, width=field_w, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=60)

        y -= 14
        _draw_text(c, label_x, y, "Prénom état civil", size=8)
        form.textfield(name=f"{prefix}_prenom_etat_civil", x=field_x, y=y - 3, width=field_w, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=60)

        y -= 14
        _draw_text(c, label_x, y, "Né(e) le", size=8)
        form.textfield(name=f"{prefix}_date_naissance", x=field_x, y=y - 3, width=75, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=20)
        _draw_text(c, field_x + 82, y, "Code postal", size=8)
        form.textfield(name=f"{prefix}_code_postal_naissance", x=field_x + 145, y=y - 3, width=60, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=10)

        y -= 14
        _draw_text(c, label_x, y, "à (ville)", size=8)
        form.textfield(name=f"{prefix}_ville_naissance", x=field_x, y=y - 3, width=field_w, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=60)

        y -= 14
        _draw_text(c, label_x, y, "Pays", size=8)
        form.textfield(name=f"{prefix}_pays_naissance", x=field_x, y=y - 3, width=field_w, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=40)

        # Address
        y -= 22
        _draw_text(c, label_x, y + 8, "Adresse principale", size=8, bold=True)
        _draw_text(c, label_x, y - 6, "N° et Rue", size=8)
        form.textfield(
            name=f"{prefix}_adresse_rue",
            x=field_x,
            y=y - 8,
            width=field_w,
            height=28,
            fillColor=colors.lightgrey,
            borderWidth=0,
            fontSize=8,
            maxlen=120,
            fieldFlags="multiline",
        )
        y -= 34
        _draw_text(c, label_x, y, "CP + Ville", size=8)
        form.textfield(name=f"{prefix}_adresse_cp_ville", x=field_x, y=y - 3, width=field_w, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=60)
        y -= 14
        _draw_text(c, label_x, y, "Pays", size=8)
        form.textfield(name=f"{prefix}_adresse_pays", x=field_x, y=y - 3, width=field_w, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=40)

        # Residence fiscale
        y -= 22
        _draw_text(c, label_x, y + 10, "Est-ce votre résidence fiscale?", size=8)
        form.checkbox(name=f"{prefix}_est_residence_fiscale_oui", x=field_x, y=y + 6, size=10, borderWidth=1, fillColor=colors.white)
        _draw_text(c, field_x + 14, y + 8, "Oui", size=8)
        form.checkbox(name=f"{prefix}_est_residence_fiscale_non", x=field_x + 55, y=y + 6, size=10, borderWidth=1, fillColor=colors.white)
        _draw_text(c, field_x + 69, y + 8, "Non", size=8)

        y -= 22
        _draw_text(c, label_x, y + 8, "Résidence fiscale (si différente)", size=8)
        form.textfield(
            name=f"{prefix}_residence_fiscale_si_differente",
            x=field_x,
            y=y - 8,
            width=field_w,
            height=26,
            fillColor=colors.lightgrey,
            borderWidth=0,
            fontSize=8,
            maxlen=140,
            fieldFlags="multiline",
        )
        y -= 34
        _draw_text(c, label_x, y, "Pays", size=8)
        form.textfield(name=f"{prefix}_residence_fiscale_pays", x=field_x, y=y - 3, width=field_w, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=40)
        return y

    y_after_left = person_block("souscripteur1", "Souscripteur 1 (personne physique)", 30, PAGE_H - 178)
    y_after_right = person_block("autre_souscripteur", "Souscripteur 1 (personne physique)", 310, PAGE_H - 178)

    # Personne morale (left)
    y_pm = min(y_after_left, y_after_right) - 18
    _draw_text(c, 30, y_pm, "Souscripteur (personne morale)", size=9, color=colors.red)
    y = y_pm - 16
    _draw_text(c, 30, y, "Raison sociale ou dénomination", size=8)
    form.textfield(name="personne_morale_raison_sociale", x=170, y=y - 3, width=260, height=12, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=120)
    y -= 14
    _draw_text(c, 30, y, "Forme juridique", size=8)
    form.textfield(name="personne_morale_forme_juridique", x=170, y=y - 3, width=120, height=12, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=40)
    y -= 14
    _draw_text(c, 30, y, "Code SIRET", size=8)
    form.textfield(name="personne_morale_siret", x=170, y=y - 3, width=120, height=12, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=20)
    y -= 14
    _draw_text(c, 30, y, "Code APE", size=8)
    form.textfield(name="personne_morale_ape", x=170, y=y - 3, width=120, height=12, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=10)
    y -= 18
    _draw_text(c, 30, y, "Représentée par (nom, prénom)", size=8, bold=True)
    form.textfield(name="personne_morale_represente_par", x=170, y=y - 3, width=260, height=12, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=120)
    y -= 14
    _draw_text(c, 30, y, "Agissant en qualité de :", size=8, bold=True)
    form.textfield(name="personne_morale_agissant_en_qualite_de", x=170, y=y - 3, width=260, height=12, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=120)

    # Beneficiaire (right)
    y_ben = y_pm
    _draw_text(c, 310, y_ben, "Bénéficiaire", size=9, color=colors.red)
    y = y_ben - 16
    label_x = 310
    field_x = 310 + 95
    field_w = PAGE_W - field_x - 30
    row_h = 12
    for label, name, width in [
        ("M. / Mme", "beneficiaire_civilite", 50),
        ("Nom", "beneficiaire_nom", field_w),
        ("Nom de jeune fille", "beneficiaire_nom_jeune_fille", field_w),
        ("Prénom d'usage", "beneficiaire_prenom_usage", field_w),
        ("Prénom état civil", "beneficiaire_prenom_etat_civil", field_w),
        ("Né(e) le", "beneficiaire_date_naissance", 75),
    ]:
        _draw_text(c, label_x, y, label, size=8)
        form.textfield(name=name, x=field_x, y=y - 3, width=width, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=80)
        if name == "beneficiaire_date_naissance":
            _draw_text(c, field_x + 82, y, "Code postal", size=8)
            form.textfield(name="beneficiaire_code_postal_naissance", x=field_x + 145, y=y - 3, width=60, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=10)
        y -= 14
    _draw_text(c, label_x, y, "à (ville)", size=8)
    form.textfield(name="beneficiaire_ville_naissance", x=field_x, y=y - 3, width=field_w, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=60)
    y -= 14
    _draw_text(c, label_x, y, "Pays", size=8)
    form.textfield(name="beneficiaire_pays_naissance", x=field_x, y=y - 3, width=field_w, height=row_h, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=40)

    # EAI block
    y_eai = y_pm - 140
    _draw_text(c, 30, y_eai, "Autocertification relative à la domiciliation fiscale (EAI)", size=9, color=colors.red)
    _draw_text(c, 30, y_eai - 12, "Etes-vous résident fiscal en France ?", size=8, bold=True)

    def eai_line(prefix: str, y0: float, label: str):
        _draw_text(c, 30, y0, label, size=8)
        form.checkbox(name=f"{prefix}_resident_france_oui", x=120, y=y0 - 4, size=10, borderWidth=1, fillColor=colors.white)
        _draw_text(c, 135, y0, "Oui", size=8)
        form.checkbox(name=f"{prefix}_resident_france_non", x=170, y=y0 - 4, size=10, borderWidth=1, fillColor=colors.white)
        _draw_text(c, 185, y0, "Non", size=8)
        _draw_text(c, 250, y0, "Pays", size=8)
        form.textfield(name=f"{prefix}_pays", x=280, y=y0 - 3, width=160, height=12, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=40)
        _draw_text(c, 460, y0, "NIF", size=8)
        form.textfield(name=f"{prefix}_nif", x=485, y=y0 - 3, width=80, height=12, fillColor=colors.lightgrey, borderWidth=0, fontSize=8, maxlen=40)

    eai_line("eai_souscripteur", y_eai - 28, "Souscripteur")
    eai_line("eai_autre_souscripteur", y_eai - 46, "Autre souscripteur")

    # FATCA block
    y_fatca = y_eai - 86
    _draw_text(c, 30, y_fatca, "Autocertification fiscale FATCA", size=9, color=colors.red)
    _draw_text(c, 30, y_fatca - 12, "Etes-vous une \"US Person\" au sens de la réglementation fiscale américaine ?", size=8, bold=True)

    def fatca_line(prefix: str, y0: float, label: str, x_label: float):
        _draw_text(c, x_label, y0, label, size=8)
        form.checkbox(name=f"{prefix}_us_person_oui", x=x_label + 85, y=y0 - 4, size=10, borderWidth=1, fillColor=colors.white)
        _draw_text(c, x_label + 100, y0, "Oui", size=8)
        form.checkbox(name=f"{prefix}_us_person_non", x=x_label + 135, y=y0 - 4, size=10, borderWidth=1, fillColor=colors.white)
        _draw_text(c, x_label + 150, y0, "Non", size=8)

    fatca_line("fatca_souscripteur", y_fatca - 28, "Souscripteur", 30)
    fatca_line("fatca_beneficiaire", y_fatca - 44, "Bénéficiaire", 30)
    fatca_line("fatca_autre_souscripteur", y_fatca - 28, "Autre souscripteur", 310)

    # General info: keep readable, spill to page 2 if needed
    info_y = y_fatca - 62
    _draw_text(
        c,
        30,
        info_y,
        "Pour chaque \"US Person\", vous devez nous communiquer le numéro d'identification fiscale TIN, ainsi qu'un formulaire W9 complété et signé (www.irs.gov/pub/irs-pdf/fw9.pdf).",
        size=7,
    )
    _draw_text(
        c,
        30,
        info_y - 10,
        "Si vous êtes une personne morale, indiquez si vous êtes une entité passive détenue à plus de 25% par un actionnaire \"US Person\" (oui/non).",
        size=7,
    )

    eai_text = (
        "A compter du 1er janvier 2016, l'entrée en vigueur de la réglementation relative aux Echanges Automatiques d'Informations (EAI) impose aux institutions financières d'identifier les éventuelles personnes résidentes fiscales à l'étranger parmi leurs clients, en vue de déclarer annuellement certains renseignements d'ordre financier aux pays ayant opté pour l'échange d'informations avec l'administration française.<br/>"
        "Vous êtes donc informé que, si vous répondez aux critères faisant de vous une personne résidente fiscale d'un pays ayant opté pour l'échange d'informations avec la France, l'institution financière est tenue de communiquer à l'administration fiscale les renseignements relatifs à votre contrat pour une année donnée et toutes les années suivantes."
    )
    fatca_text = (
        "A compter du 1er juillet 2014, l'Accord intergouvernemental signé le 14 novembre 2013 entre le Gouvernement français et le Gouvernement américain impose aux institutions financières d'identifier les éventuels contribuables américains (« US Person ») parmi leurs clients, en vue de déclarer annuellement certains renseignements d'ordre financier.<br/>"
        "Vous êtes donc informé que, si vous répondez aux critères faisant de vous un contribuable américain (« US Person »), notamment si vous êtes citoyen ou résident américain, l'institution financière est tenue de communiquer chaque année à l'administration fiscale française les données relatives à votre contrat, dans la mesure où il répond aux conditions définies par cet accord."
    )

    general_w = PAGE_W - 60
    title_h = 12
    gap = 6
    para_size = 6.3
    para_leading = 7.2
    eai_h = _measure_paragraph_height(eai_text, general_w, para_size, para_leading)
    fatca_h = _measure_paragraph_height(fatca_text, general_w, para_size, para_leading)
    needed = title_h + gap + eai_h + 10 + title_h + gap + fatca_h

    top_y = info_y - 26
    bottom_margin = 30
    if top_y - needed < bottom_margin:
        c.showPage()
        top_y = PAGE_H - 50

    y = top_y
    _draw_text(c, 30, y, "Informations générales sur EAI", size=9, color=colors.red)
    y -= title_h
    _draw_paragraph(c, 30, y, eai_text, w=general_w, size=para_size, leading=para_leading)
    y -= eai_h + 10
    _draw_text(c, 30, y, "Informations générales sur FATCA", size=9, color=colors.red)
    y -= title_h
    _draw_paragraph(c, 30, y, fatca_text, w=general_w, size=para_size, leading=para_leading)

    c.showPage()
    c.save()
    return buffer.getvalue()


@lru_cache(maxsize=1)
def get_template_pdf_bytes() -> bytes:
    return _build_template_pdf_bytes()


def fill_template_pdf_bytes(template_pdf: bytes, data: dict[str, Any]) -> bytes:
    reader = PdfReader(io.BytesIO(template_pdf))
    writer = PdfWriter(clone_from=reader)
    writer.set_need_appearances_writer()

    field_values = _coerce_field_values(data)
    for page in writer.pages:
        if "/Annots" in page:
            writer.update_page_form_field_values(page, field_values)

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def build_fatca_eai_autocertification_pdf_bytes(data: dict[str, Any]) -> bytes:
    return fill_template_pdf_bytes(get_template_pdf_bytes(), data)

