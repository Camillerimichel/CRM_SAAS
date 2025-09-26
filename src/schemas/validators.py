from datetime import datetime

def format_date(v):
    """
    Convertit un datetime en str au format YYYY-MM-DD.
    Si v n'est pas un datetime, le renvoie tel quel.
    """
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d")
    return v
