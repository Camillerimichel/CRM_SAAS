from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable, Tuple

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _load_project_env() -> None:
    """Charge le .env racine sans écraser les variables déjà injectées par le service."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


_load_project_env()


def _str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _smtp_settings() -> tuple[str | None, str, str | None, int, str | None, str | None, bool, bool]:
    host = os.getenv("SMTP_HOST")
    sender = os.getenv("SMTP_FROM") or os.getenv("SMTP_USER") or os.getenv("SMTP_USERNAME")
    sender_name = os.getenv("SMTP_FROM_NAME")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD") or os.getenv("SMTP_PASS")
    use_ssl = _str_to_bool(os.getenv("SMTP_USE_SSL"), default=(port == 465))
    use_tls = _str_to_bool(os.getenv("SMTP_USE_TLS"), default=not use_ssl)
    return host, sender or "", sender_name, port, username, password, use_ssl, use_tls


def send_email(to_email: str, subject: str, body: str) -> bool:
    """Envoi SMTP simple (texte) basé sur variables d'environnement. Retourne True si OK."""
    host, sender, sender_name, port, username, password, use_ssl, use_tls = _smtp_settings()
    if not host:
        logger.warning("SMTP désactivé : variable SMTP_HOST absente.")
        return False

    if not sender:
        logger.warning("SMTP désactivé : aucun expéditeur défini (SMTP_FROM ou SMTP_USERNAME).")
        return False

    message = EmailMessage()
    message["From"] = f"{sender_name} <{sender}>" if sender_name else sender
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


def send_email_with_attachments(
    to_email: str,
    subject: str,
    body: str,
    attachments: Iterable[tuple[str, bytes, str]] = (),
) -> tuple[bool, str | None]:
    """Envoi SMTP avec pièces jointes binaires.

    Retourne (succès, erreur) afin que l'UI puisse afficher la vraie cause.
    """
    host, sender, sender_name, port, username, password, use_ssl, use_tls = _smtp_settings()
    if not host:
        logger.warning("SMTP désactivé : variable SMTP_HOST absente.")
        return False, "SMTP_HOST absent."

    if not sender:
        logger.warning("SMTP désactivé : aucun expéditeur défini (SMTP_FROM ou SMTP_USERNAME).")
        return False, "Aucun expéditeur SMTP défini."

    message = EmailMessage()
    message["From"] = f"{sender_name} <{sender}>" if sender_name else sender
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body or "")

    for filename, file_bytes, mime_type in attachments:
        maintype, subtype = (mime_type or "application/pdf").split("/", 1) if "/" in (mime_type or "") else ("application", "pdf")
        message.add_attachment(file_bytes, maintype=maintype, subtype=subtype, filename=Path(filename).name)

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
    except smtplib.SMTPAuthenticationError as exc:
        logger.exception("Échec d'authentification SMTP pour %s", to_email)
        detail = exc.smtp_error.decode("utf-8", errors="replace") if isinstance(exc.smtp_error, (bytes, bytearray)) else str(exc.smtp_error or exc)
        return False, f"Authentification SMTP refusée ({getattr(exc, 'smtp_code', 'auth')}): {detail}"
    except Exception as exc:
        logger.exception("Échec de l'envoi de l'email à %s", to_email)
        return False, f"Erreur SMTP: {exc}"

    return True, None
