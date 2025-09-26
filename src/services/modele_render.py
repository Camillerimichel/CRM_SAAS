from sqlalchemy.orm import Session
from src.models.modele_document import ModeleDocument


def render_modele(db: Session, modele_id: int, data: dict) -> dict:
    m = db.query(ModeleDocument).filter(ModeleDocument.id == modele_id).first()
    if not m:
        raise ValueError("modele_not_found")

    def _apply(t: str | None):
        if not t:
            return t
        out = t
        for k, v in (data or {}).items():
            out = out.replace("{{" + str(k) + "}}", str(v))
        return out

    return {"objet": _apply(m.objet), "contenu": _apply(m.contenu)}

