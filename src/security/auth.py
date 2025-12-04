import os
from typing import Optional

import bcrypt
from itsdangerous import URLSafeSerializer, URLSafeTimedSerializer, BadSignature, BadTimeSignature


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def _serializer() -> URLSafeSerializer:
    secret = os.getenv("SECRET_KEY") or "dev-secret-key-change-me"
    return URLSafeSerializer(secret_key=secret, salt="auth-token")


def _timed_serializer() -> URLSafeTimedSerializer:
    secret = os.getenv("SECRET_KEY") or "dev-secret-key-change-me"
    return URLSafeTimedSerializer(secret_key=secret, salt="reset-token")


def encode_token(user_id: int, user_type: str, societe_id: Optional[int]) -> str:
    payload = {"uid": user_id, "utype": user_type, "sid": societe_id}
    return _serializer().dumps(payload)


def decode_token(token: str) -> Optional[dict]:
    try:
        return _serializer().loads(token)
    except BadSignature:
        return None


def encode_reset_token(user_id: int, user_type: str, ttl_seconds: int = 3600) -> str:
    payload = {"uid": user_id, "utype": user_type}
    # Timed serializer embarque l'horodatage; la limite est appliquée à la lecture.
    return _timed_serializer().dumps(payload, salt="pwd-reset")


def decode_reset_token(token: str, max_age: int = 3600) -> Optional[dict]:
    try:
        return _timed_serializer().loads(token, salt="pwd-reset", max_age=max_age)
    except (BadSignature, BadTimeSignature):
        return None
