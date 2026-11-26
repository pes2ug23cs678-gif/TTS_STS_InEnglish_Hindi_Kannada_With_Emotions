
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend_api.db import get_db
from backend_api.main import app
from backend_api.models import Base, User
from backend_api.auth import hash_password, create_token


TEST_DATABASE_URL = "sqlite://"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(
    bind=test_engine,
    autoflush=False,
    autocommit=False,
)


@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=test_engine)
    db = TestingSessionLocal()

    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db_session):
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("SecurePass123"),
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_register_user(client):
    response = client.post(
        "/register",
        json={
            "name": "New User",
            "email": "new@example.com",
            "password": "SecurePass123",
        },
    )

    assert response.status_code == 201
    assert response.json()["name"] == "New User"
    assert response.json()["email"] == "new@example.com"
    assert "password_hash" not in response.json()


def test_register_duplicate_email(client, test_user):
    response = client.post(
        "/register",
        json={
            "name": "Another User",
            "email": "test@example.com",
            "password": "SecurePass123",
        },
    )

    assert response.status_code == 409


def test_register_invalid_password(client):
    response = client.post(
        "/register",
        json={
            "name": "New User",
            "email": "new@example.com",
            "password": "short",
        },
    )

    assert response.status_code == 422


def test_login_success(client, test_user):
    response = client.post(
        "/login",
        data={
            "username": "test@example.com",
            "password": "SecurePass123",
        },
    )

    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    assert response.json()["access_token"]


def test_login_wrong_password(client, test_user):
    response = client.post(
        "/login",
        data={
            "username": "test@example.com",
            "password": "WrongPass123",
        },
    )

    assert response.status_code == 401


def test_history_requires_authentication(client):
    response = client.get("/history")

    assert response.status_code == 401


def test_translate_requires_authentication(client):
    response = client.post(
        "/translate",
        files={"file": ("test.wav", b"audio", "audio/wav")},
        data={"output_lang": "hindi", "gender": "female"},
    )

    assert response.status_code == 401


def test_translate_rejects_unsupported_language(client, test_user):
    token = create_token(test_user.id)

    response = client.post(
        "/translate",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("test.wav", b"audio", "audio/wav")},
        data={"output_lang": "telugu", "gender": "female"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "output_lang must be english, hindi or kannada"


def test_translate_rejects_non_audio_extension(client, test_user):
    token = create_token(test_user.id)

    response = client.post(
        "/translate",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("license.txt", b"not audio", "text/plain")},
        data={"output_lang": "hindi", "gender": "female"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported audio file format"


def test_audio_endpoint_requires_authentication(client):
    response = client.get("/audio/1/output")

    assert response.status_code == 401
