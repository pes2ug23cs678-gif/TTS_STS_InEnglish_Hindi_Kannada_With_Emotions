
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import HTTPException
from unittest.mock import Mock

from backend_api.auth import (
    ALGORITHM,
    create_token,
    get_current_user,
    hash_password,
    verify_password,
)
from backend_api.config import SECRET_KEY
from backend_api.models import User
from backend_api.schemas import UserCreate


def test_password_hash_and_verify():
    password = "SecurePass123"
    hashed = hash_password(password)

    assert hashed != password
    assert verify_password(password, hashed)
    assert not verify_password("WrongPass123", hashed)


def test_password_hashes_are_salted():
    password = "SecurePass123"

    first_hash = hash_password(password)
    second_hash = hash_password(password)

    assert first_hash != second_hash
    assert verify_password(password, first_hash)
    assert verify_password(password, second_hash)


def test_verify_password_rejects_password_over_72_bytes():
    assert not verify_password("a" * 73, "invalid-hash")


def test_verify_password_rejects_invalid_hash():
    assert not verify_password("SecurePass123", "invalid-hash")


def test_create_token_contains_user_id():
    token = create_token(123)

    payload = jwt.decode(
        token,
        SECRET_KEY,
        algorithms=[ALGORITHM],
    )

    assert payload["sub"] == "123"
    assert payload["exp"] > datetime.now(timezone.utc).timestamp()


def test_get_current_user_returns_existing_user():
    user = User(id=3, name="Test User", email="test@example.com",
                password_hash="hashed")
    db = Mock()
    db.get.return_value = user
    token = create_token(3)

    result = get_current_user(token=token, db=db)

    assert result is user
    db.get.assert_called_once_with(User, 3)


def test_get_current_user_rejects_expired_token():
    expired_token = jwt.encode(
        {
            "sub": "3",
            "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
        },
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    db = Mock()

    with pytest.raises(HTTPException) as exc:
        get_current_user(token=expired_token, db=db)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid or expired token"
    db.get.assert_not_called()


def test_get_current_user_rejects_invalid_token():
    db = Mock()

    with pytest.raises(HTTPException) as exc:
        get_current_user(token="invalid-token", db=db)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid or expired token"


def test_get_current_user_rejects_missing_user():
    db = Mock()
    db.get.return_value = None
    token = create_token(999)

    with pytest.raises(HTTPException) as exc:
        get_current_user(token=token, db=db)

    assert exc.value.status_code == 401
    assert exc.value.detail == "User not found"


def test_user_schema_trims_name_and_normalizes_email():
    user = UserCreate(
        name="  Test User  ",
        email="TEST@example.com",
        password="SecurePass123",
    )

    assert user.name == "Test User"
    assert str(user.email) == "TEST@example.com"


def test_user_schema_rejects_short_password():
    with pytest.raises(ValueError):
        UserCreate(
            name="Test User",
            email="test@example.com",
            password="short",
        )


def test_user_schema_rejects_whitespace_only_name():
    with pytest.raises(ValueError):
        UserCreate(
            name="   ",
            email="test@example.com",
            password="SecurePass123",
        )


def test_user_schema_rejects_whitespace_only_password():
    with pytest.raises(ValueError):
        UserCreate(
            name="Test User",
            email="test@example.com",
            password="        ",
        )


def test_user_schema_rejects_password_over_72_bytes():
    with pytest.raises(ValueError):
        UserCreate(
            name="Test User",
            email="test@example.com",
            password="a" * 73,
        )
