from pydantic import BaseModel, field_validator

class SupportSchema(BaseModel):
    id: int
    code_isin: str | None = None
    nom: str | None = None
    cat_gene: str | None = None
    cat_principale: str | None = None
    cat_det: str | None = None
    cat_geo: str | None = None
    promoteur: str | None = None
    taux_retro: float | None = None
    SRRI: int | None = None

    class Config:
        from_attributes = True

    @field_validator("taux_retro", mode="before")
    @classmethod
    def parse_taux(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            v = v.strip().replace(",", ".")
            if v.startswith("00."):
                v = v[1:]
        try:
            return float(v)
        except Exception:
            return None


# Création (API → DB)
class SupportCreateSchema(BaseModel):
    code_isin: str
    nom: str
    cat_gene: str | None = None
    cat_principale: str | None = None
    cat_det: str | None = None
    cat_geo: str | None = None
    promoteur: str | None = None
    taux_retro: float | None = None
    SRRI: int | None = None
