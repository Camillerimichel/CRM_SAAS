from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

logger = logging.getLogger(__name__)


def _str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def send_email(to_email: str, subject: str, body: str) -> bool:
    """Envoi SMTP simple (texte) basé sur variables d'environnement. Retourne True si OK."""
    host = os.getenv("SMTP_HOST")
    if not host:
        logger.warning("SMTP désactivé : variable SMTP_HOST absente.")
        return False

    sender = os.getenv("SMTP_FROM", os.getenv("SMTP_USERNAME"))
    if not sender:
        logger.warning("SMTP désactivé : aucun expéditeur défini (SMTP_FROM ou SMTP_USERNAME).")
        return False

    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    use_ssl = _str_to_bool(os.getenv("SMTP_USE_SSL"), default=False)
    use_tls = _str_to_bool(os.getenv("SMTP_USE_TLS"), default=not use_ssl)

    message = EmailMessage()
    message["From"] = sender
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body)

    try:
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=15) as server:
                if username and password:
                    server.login(username, password)
                server.send_message(message)
        else:
            with smtplib.SMTP(host, port, timeout=15) as server:
                if use_tls:
                    context = ssl.create_default_context()
                    server.starttls(context=context)
                if username and password:
                    server.login(username, password)
                server.send_message(message)
    except Exception:  # pragma: no cover
        logger.exception("Échec de l'envoi de l'email à %s", to_email)
        return False

    return True
