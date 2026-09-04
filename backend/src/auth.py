"""
Authentication module for JWT-based token authentication.

Provides:
- Password hashing and verification
- JWT token generation and validation
- Demo user database with roles
"""

import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
from passlib.context import CryptContext
from pydantic import BaseModel
from config import Config
from rbac import is_valid_role

# ============================================================
# CONFIGURATION
# ============================================================

# JWT configuration
SECRET_KEY = Config.JWT_SECRET_KEY
ALGORITHM = "HS256"
TOKEN_EXPIRY_MINUTES = Config.TOKEN_EXPIRY_MINUTES

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ============================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================

class LoginRequest(BaseModel):
    """Login request payload."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response payload."""
    access_token: str
    token_type: str
    username: str
    role: str


class TokenPayload(BaseModel):
    """JWT token payload."""
    username: str
    role: str
    exp: datetime


# ============================================================
# DEMO USER DATABASE
# ============================================================

# Demo users with hashed passwords
# Generated using: pwd_context.hash("password123")
DEMO_USERS: Dict[str, Dict[str, str]] = {
    "Tony": {
        "password_hash": "$2b$12$S78xUErx08z0iRjPpkx1zeIynrKcI3ogbxhPnW2resOKlc0.qnBTq",
        "role": "engineering"
    },
    "Peter": {
        "password_hash": "$2b$12$gvFzoXE2KNJWngtJHdCpLemCW8EeVdbAHJnyjp8zmlMVNszXqvaCi",
        "role": "engineering"
    },
    "Natasha": {
        "password_hash": "$2b$12$G0z/o6H7AZPJtn9FJGNaSeeyosMFIr3C8vXMKvGYVEeiae7tmp13C",
        "role": "hr"
    },
    "Shashank": {
        "password_hash": "$2b$12$JBljd6jkt4ScishJXcp/p.vkYfR2GWw5LxVEdCkUDA/K0V9O7H66u",
        "role": "c_level"
    }
}


# ============================================================
# PASSWORD UTILITIES
# ============================================================

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


# ============================================================
# JWT TOKEN UTILITIES
# ============================================================

def create_access_token(username: str, role: str) -> str:
    """
    Create a JWT access token.
    
    Args:
        username: User's username
        role: User's role (hr, engineering, c_level, etc.)
    
    Returns:
        Encoded JWT token
    """
    if not is_valid_role(role):
        raise ValueError(f"Invalid role: {role}")
    if not SECRET_KEY or SECRET_KEY == "change-this-secret-in-production":
        raise ValueError("JWT_SECRET_KEY must be configured")

    now = datetime.now(timezone.utc)
    expiry = now + timedelta(minutes=TOKEN_EXPIRY_MINUTES)
    
    payload = {
        "username": username,
        "role": role,
        "exp": expiry,
        "iat": now,
    }
    
    encoded = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return encoded


def decode_access_token(token: str) -> Optional[TokenPayload]:
    """
    Decode and validate a JWT access token.
    
    Args:
        token: JWT token string
    
    Returns:
        TokenPayload if valid, None if invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("username")
        role = payload.get("role")
        exp = payload.get("exp")
        
        if username is None or role is None:
            return None
        
        if not is_valid_role(role):
            return None

        return TokenPayload(
            username=username,
            role=role,
            exp=datetime.fromtimestamp(exp, tz=timezone.utc)
        )
    except jwt.ExpiredSignatureError:
        return None  # Token expired
    except jwt.InvalidTokenError:
        return None  # Invalid token


# ============================================================
# AUTHENTICATION FUNCTIONS
# ============================================================

def authenticate_user(username: str, password: str) -> Optional[Dict[str, str]]:
    """
    Authenticate a user with username and password.
    
    Args:
        username: User's username
        password: User's password (plain text)
    
    Returns:
        Dict with username and role if successful, None if invalid credentials
    """
    user = DEMO_USERS.get(username)
    
    if not user:
        return None  # User not found
    
    if not verify_password(password, user["password_hash"]):
        return None  # Password mismatch
    
    return {
        "username": username,
        "role": user["role"]
    }


