from __future__ import annotations

import unicodedata
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session


QUESTION_FIELDS = (
    "env_importance",
    "env_ges_reduc",
    "soc_droits_humains",
    "soc_parite",
    "gov_transparence",
    "gov_controle_ethique",
)


def _norm_token(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip().lower()
    if not raw:
        return ""
    normalized = unicodedata.normalize("NFKD", raw)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _answer_factor(value: Any) -> float:
    token = _norm_token(value)
    if token in {"oui", "yes", "true", "1"}:
        return 1.0
    if token in {"indifferent", "indifferent", "neutre", "neutral"}:
        return 0.5
    if token in {"non", "no", "false", "0"}:
        return 0.0
    return 0.0


def _labelize(value: str) -> str:
    if not value:
        return ""
    return str(value).replace("_", " ").strip().capitalize()


def get_esg_top_metrics(db: Session, client_id: int, top_n: int = 5) -> list[dict[str, Any]]:
    """Return top ESG metrics for a client based on sensitivity answers and weights."""
    row = db.execute(
        text(
            """
            SELECT *
            FROM esg_questionnaire
            WHERE client_ref = :r
            ORDER BY COALESCE(updated_at, saisie_at, created_at, id) DESC
            LIMIT 1
            """
        ),
        {"r": str(client_id)},
    ).fetchone()
    if not row:
        return []

    q = row._mapping if hasattr(row, "_mapping") else row
    answers = {k: q.get(k) if hasattr(q, "get") else None for k in QUESTION_FIELDS}

    qid = q.get("id") if hasattr(q, "get") else None
    selected_keys: set[str] = set()
    if qid is not None:
        excl_rows = db.execute(
            text(
                """
                SELECT o.id, o.code
                FROM esg_questionnaire_exclusion qe
                LEFT JOIN esg_exclusion_option o ON o.id = qe.option_id
                WHERE qe.questionnaire_id = :q
                """
            ),
            {"q": int(qid)},
        ).fetchall()
        ind_rows = db.execute(
            text(
                """
                SELECT o.id, o.code
                FROM esg_questionnaire_indicator qi
                LEFT JOIN esg_indicator_option o ON o.id = qi.option_id
                WHERE qi.questionnaire_id = :q
                """
            ),
            {"q": int(qid)},
        ).fetchall()
        for rid, code in (excl_rows or []) + (ind_rows or []):
            if rid is not None:
                selected_keys.add(_norm_token(rid))
            if code:
                selected_keys.add(_norm_token(code))

    weight_rows = db.execute(
        text(
            """
            SELECT dimension, dimension_key, metric_key, weight
            FROM esg_sensitivity_weights
            WHERE active = 1
            """
        )
    ).fetchall()

    scores: dict[str, float] = {}
    for row_w in weight_rows or []:
        m = row_w._mapping if hasattr(row_w, "_mapping") else row_w
        dimension = str(m.get("dimension") or "").strip().lower()
        dimension_key = str(m.get("dimension_key") or "").strip()
        metric_key = str(m.get("metric_key") or "").strip()
        weight = float(m.get("weight") or 0.0)
        if not metric_key or weight <= 0:
            continue

        factor = 0.0
        if dimension == "question":
            factor = _answer_factor(answers.get(dimension_key))
        elif dimension in {"exclusion", "indicator"}:
            if _norm_token(dimension_key) in selected_keys:
                factor = 1.0

        if factor <= 0:
            continue
        scores[metric_key] = scores.get(metric_key, 0.0) + (weight * factor)

    if not scores:
        return []

    stats_row = db.execute(text("SELECT * FROM esg_metric_averages LIMIT 1")).fetchone()
    stats = stats_row._mapping if stats_row is not None and hasattr(stats_row, "_mapping") else None

    def _metric_stats(metric_key: str) -> tuple[float | None, float | None]:
        if not stats:
            return None, None
        return (
            stats.get(f"{metric_key}_avg"),
            stats.get(f"{metric_key}_coverage_pct"),
        )

    scored = []
    for metric_key, score in scores.items():
        avg_val, coverage_pct = _metric_stats(metric_key)
        if coverage_pct is not None and coverage_pct <= 60.0:
            continue
        quality = float(coverage_pct) / 100.0 if coverage_pct is not None else 1.0
        scored.append(
            {
                "metric_key": metric_key,
                "score": float(score),
                "quality": quality,
                "avg_value": avg_val,
                "coverage_pct": coverage_pct,
                "score_adjusted": float(score) * quality,
            }
        )

    ranked = sorted(scored, key=lambda item: (-item["score_adjusted"], item["metric_key"]))[: max(1, top_n)]
    metric_keys = [item["metric_key"] for item in ranked]

    label_rows = db.execute(
        text(
            """
            SELECT metric_key, label_fr, unit
            FROM esg_metric_labels
            WHERE metric_key IN :keys
            """
        ).bindparams(bindparam("keys", expanding=True)),
        {"keys": metric_keys},
    ).fetchall()
    labels = {}
    for row_l in label_rows or []:
        m = row_l._mapping if hasattr(row_l, "_mapping") else row_l
        labels[m.get("metric_key")] = {
            "label": m.get("label_fr"),
            "unit": m.get("unit"),
        }

    results = []
    for item in ranked:
        metric_key = item["metric_key"]
        label_data = labels.get(metric_key) or {}
        label = label_data.get("label") or _labelize(metric_key)
        results.append(
            {
                "metric_key": metric_key,
                "label": label,
                "unit": label_data.get("unit"),
                "score": round(item["score"], 4),
                "score_adjusted": round(item["score_adjusted"], 4),
                "avg_value": item["avg_value"],
                "coverage_pct": item["coverage_pct"],
            }
        )
    return results
