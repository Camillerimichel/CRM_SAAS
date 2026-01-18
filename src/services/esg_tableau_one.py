from datetime import date as _date, datetime

from sqlalchemy import bindparam, text


TABLEAU_ONE_CATEGORIES = {
    "climate": "Indicateurs climatiques",
    "biodiversity": "Indicateurs biodiversité",
    "water": "Indicateurs eau",
    "waste": "Indicateurs déchets",
    "social": "Indicateurs sociaux & gouvernance",
}

TABLEAU_ONE_CATEGORY_ORDER = [
    "climate",
    "biodiversity",
    "water",
    "waste",
    "social",
]

TABLEAU_ONE_INDICATORS = [
    {
        "key": "scope1and2carbonintensity",
        "label": "Émissions de GES de niveau 1 + 2",
        "category": "climate",
        "section": "Émissions de gaz à effet de serre",
        "measure": "tonnes GHG / $M revenu",
        "unit": "tCO2e",
        "decimals": 2,
    },
    {
        "key": "scope3carbonintensity",
        "label": "Émissions de GES de niveau 3",
        "category": "climate",
        "section": "Émissions de gaz à effet de serre",
        "measure": "tonnes GHG / $M revenu",
        "unit": "tCO2e",
        "decimals": 2,
    },
    {
        "key": "carbontrend",
        "label": "Empreinte carbone (%)",
        "category": "climate",
        "section": "Empreinte carbone",
        "measure": "Evolution des émissions de CO₂ (réduction/stabilité/augmentation)",
        "unit": "%",
        "decimals": 2,
        "transform": lambda value: value * 100,
    },
    {
        "key": "ghgintensityvalue",
        "label": "Intensité de GES des sociétés bénéficiaires",
        "category": "climate",
        "section": "Intensité GES",
        "measure": "tCO2e par million d’euros de chiffre d’affaires",
        "unit": "tCO2e/M€",
        "decimals": 2,
    },
    {
        "key": "exposuretofossilfuels",
        "label": "Part d’investissement dans les combustibles fossiles",
        "category": "climate",
        "section": "Combustibles fossiles",
        "measure": "Part en % des sociétés actives dans ce secteur",
        "unit": "%",
        "decimals": 2,
        "transform": lambda value: value * 100,
    },
    {
        "key": "renewableenergy",
        "label": "Part d’énergie non renouvelable",
        "category": "climate",
        "section": "Énergie",
        "measure": "Part % de l’énergie non renouvelable sur le total",
        "unit": "%",
        "decimals": 2,
        "transform": lambda value: (1 - value) * 100,
    },
    {
        "key": "biodiversity",
        "label": "Activités ayant une incidence négative sur des zones sensibles sur le plan de la biodiversité",
        "category": "biodiversity",
        "section": "Biodiversité",
        "measure": "",
        "unit": "",
        "decimals": 2,
    },
    {
        "key": "waterefficiency",
        "label": "Water efficiency",
        "category": "water",
        "section": "Eau — Facteurs ESG corporate",
        "measure": "Milliers de m³ d’eau douce utilisés par million € de chiffre d’affaires",
        "unit": "score",
        "decimals": 2,
    },
    {
        "key": "hazardouswaste",
        "label": "Ratio de déchets dangereux et de déchets radioactifs",
        "category": "waste",
        "section": "Déchets",
        "measure": "Tonnes de déchets dangereux et de déchets radioactifs produites / EVIC",
        "unit": "tonnes de déchets générés/EVIC",
        "decimals": 2,
    },
    {
        "key": "violationsungc",
        "label": "Violations des principes du pacte mondial des Nations unies et des principes directeurs de l’OCDE pour les entreprises multinationales",
        "category": "social",
        "section": "Les questions sociales et de personnel",
        "measure": "Part d’investissement dans des sociétés qui ont participé à des violations des principes du Pacte mondial des Nations unies ou des principes directeurs de l’OCDE à l’intention des entreprises multinationales",
        "unit": "%",
        "decimals": 2,
        "transform": lambda value: value * 100,
    },
    {
        "key": "processesungc",
        "label": "Absence de processus et de mécanismes de conformité permettant de contrôler le respect des principes du Pacte mondial des Nations unies et des principes directeurs de l’OCDE à l’intention des entreprises multinationales",
        "category": "social",
        "section": "Les questions sociales et de personnel",
        "measure": "Part d’investissement dans des sociétés qui n’ont pas de politique de contrôle du respect des principes du Pacte mondial des Nations unies ou des principes directeurs de l’OCDE à l’intention des entreprises multinationales, ni de mécanismes de traitement des plaintes ou des différents permettant de remédier à de telles violations",
        "unit": "%",
        "decimals": 2,
        "transform": lambda value: value * 100,
    },
    {
        "key": "genderpaygap",
        "label": "Écart de rémunération entre hommes et femmes non corrigé",
        "category": "social",
        "section": "Les questions sociales et de personnel",
        "measure": "Écart de rémunération moyen non corrigé entre les hommes et les femmes au sein des sociétés bénéficiaires des investissements",
        "unit": "%",
        "decimals": 2,
        "transform": lambda value: value * 100,
    },
    {
        "key": "boardgenderdiversity",
        "label": "Mixité au sein des organes de gouvernance",
        "category": "social",
        "section": "Les questions sociales et de personnel",
        "measure": "Ratio femmes/hommes moyen dans les organes de gouvernance des sociétés concernées, en pourcentage du nombre total de membres",
        "unit": "%",
        "decimals": 2,
        "transform": lambda value: value * 100,
    },
    {
        "key": "controversialweapons",
        "label": "Exposition à des armes controversées (mines antipersonnel, armes à sous-munitions, armes chimiques ou armes biologiques)",
        "category": "social",
        "section": "Les questions sociales et de personnel",
        "measure": "Part d’investissement dans des sociétés qui participent à la fabrication ou à la vente d’armes controversées",
        "unit": "%",
        "decimals": 2,
        "transform": lambda value: value * 100,
    },
]


def _parse_esg_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"na", "n/a", "null", "none", "-"}:
        return None
    cleaned = raw.replace(" ", "").replace("%", "")
    if cleaned.count(",") > 0 and cleaned.count(".") == 0:
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except Exception:
        return None


def _format_number(value, decimals=2):
    if value is None:
        return "—"
    try:
        num = float(value)
    except Exception:
        return "—"
    fmt = f"{{:,.{decimals}f}}"
    return fmt.format(num).replace(",", " ")


def compute_tableau_one(db):
    last_date = db.execute(text("SELECT MAX(date) FROM mariadb_historique_support_w")).scalar()
    if not last_date:
        return {"as_of_date": None, "total_valo": 0.0, "items": [], "categories": []}
    if isinstance(last_date, (datetime, _date)):
        last_date_str = last_date.strftime("%Y-%m-%d")
    else:
        last_date_str = str(last_date)[:10]

    supports_rows = db.execute(
        text(
            """
            SELECT s.code_isin AS isin, SUM(h.valo) AS valo
            FROM mariadb_support s
            LEFT JOIN mariadb_historique_support_w h
              ON h.id_support = s.id AND DATE(h.date) = :last_date
            GROUP BY s.code_isin
            """
        ),
        {"last_date": last_date},
    ).mappings().all()

    weights = {}
    total_valo = 0.0
    for row in supports_rows or []:
        isin = (row.get("isin") or "").strip()
        if not isin:
            continue
        try:
            valo = float(row.get("valo") or 0.0)
        except Exception:
            valo = 0.0
        if valo <= 0:
            continue
        weights[isin] = valo
        total_valo += valo

    if not weights:
        return {"as_of_date": last_date_str, "total_valo": 0.0, "items": [], "categories": []}

    cols = ", ".join(f"`{indicator['key']}`" for indicator in TABLEAU_ONE_INDICATORS)
    esg_rows = db.execute(
        text(
            f"""
            SELECT isin, {cols}
            FROM donnees_esg_etendu
            WHERE isin IN :isins
            """
        ).bindparams(bindparam("isins", expanding=True)),
        {"isins": list(weights.keys())},
    ).mappings().all()

    esg_by_isin = {}
    for row in esg_rows or []:
        isin = (row.get("isin") or "").strip()
        if not isin:
            continue
        esg_by_isin[isin] = row

    items = []
    for indicator in TABLEAU_ONE_INDICATORS:
        key = indicator["key"]
        weighted_sum = 0.0
        weight_total = 0.0
        transform = indicator.get("transform")
        for isin, weight in weights.items():
            row = esg_by_isin.get(isin)
            if not row:
                continue
            value = _parse_esg_number(row.get(key))
            if value is None:
                continue
            if transform:
                value = transform(value)
            weighted_sum += value * weight
            weight_total += weight
        avg_value = None
        if weight_total > 0:
            avg_value = weighted_sum / weight_total
        coverage_pct = (weight_total / total_valo * 100.0) if total_valo > 0 else 0.0
        items.append({
            "key": key,
            "label": indicator["label"],
            "category": indicator["category"],
            "category_label": TABLEAU_ONE_CATEGORIES.get(indicator["category"], indicator["category"]),
            "section": indicator.get("section") or indicator["label"],
            "measure": indicator.get("measure") or "",
            "unit": indicator.get("unit") or "",
            "value": avg_value,
            "coverage_pct": coverage_pct,
            "value_fmt": _format_number(avg_value, indicator.get("decimals", 2)),
            "coverage_fmt": _format_number(coverage_pct, 1) + " %",
        })

    categories = []
    for category_key in TABLEAU_ONE_CATEGORY_ORDER:
        label = TABLEAU_ONE_CATEGORIES.get(category_key, category_key)
        category_items = [item for item in items if item["category"] == category_key]
        if category_items:
            categories.append({
                "key": category_key,
                "label": label,
                "items": category_items,
            })

    return {
        "as_of_date": last_date_str,
        "total_valo": total_valo,
        "items": items,
        "categories": categories,
    }
