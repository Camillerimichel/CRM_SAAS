"""Shared ESG grade helpers."""

from __future__ import annotations

ESG_GRADE_TO_NUM: dict[str, int] = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
}

ESG_NUM_TO_GRADE: dict[int, str] = {v: k for k, v in ESG_GRADE_TO_NUM.items()}


def esg_letter_to_num(raw) -> int | None:
    if raw is None:
        return None
    code = str(raw).strip().upper()[:1]
    return ESG_GRADE_TO_NUM.get(code)


def esg_grade_from_letters(note_e, note_s, note_g) -> str | None:
    """Return the ESG grade using the 50/30/20 weighted rule.

    Inputs can be letters A..G or any string whose first character is A..G.
    """
    num_e = esg_letter_to_num(note_e)
    num_s = esg_letter_to_num(note_s)
    num_g = esg_letter_to_num(note_g)
    if num_e is None and num_s is None and num_g is None:
        return None

    score = 0.0
    weight_sum = 0.0
    if num_e is not None:
        score += num_e * 0.5
        weight_sum += 0.5
    if num_s is not None:
        score += num_s * 0.3
        weight_sum += 0.3
    if num_g is not None:
        score += num_g * 0.2
        weight_sum += 0.2
    if weight_sum <= 0:
        return None

    score /= weight_sum
    grade_num = int(round(score))
    if grade_num < 1:
        grade_num = 1
    if grade_num > 7:
        grade_num = 7
    return ESG_NUM_TO_GRADE.get(grade_num)
