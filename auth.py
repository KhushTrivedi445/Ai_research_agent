"""
auth.py
Simple username/password authentication with JWT session tokens.

Users are stored in a local sqlite database (separate table from chat
history). Passwords are hashed with bcrypt — never stored in plain text.
On successful login, a JWT is issued and kept in Streamlit's session
state to represent "you are logged in" for the rest of the session.
"""

import os
import sqlite3
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

DB_PATH = "chat_history.db"  # same file as chat checkpoints, different table

# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------
def _get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_user_table():
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_user_table()

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------
def username_exists(username: str) -> bool:
    conn = _get_conn()
    row = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row is not None


def create_user(username: str, password: str) -> tuple[bool, str]:
    """Returns (success, message)."""
    username = username.strip()
    if not username or not password:
        return False, "Username and password are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if username_exists(username):
        return False, "That username is already taken."

    conn = _get_conn()
    conn.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (username, _hash_password(password), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return True, "Account created successfully."


def authenticate_user(username: str, password: str) -> tuple[bool, str]:
    """Returns (success, message)."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT password_hash FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()

    if not row:
        return False, "No account with that username."
    if not _verify_password(password, row[0]):
        return False, "Incorrect password."
    return True, "Login successful."


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------
def generate_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> str | None:
    """Returns the username if the token is valid and unexpired, else None."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None