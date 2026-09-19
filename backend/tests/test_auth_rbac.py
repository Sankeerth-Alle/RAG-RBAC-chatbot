from fastapi.testclient import TestClient

from auth import authenticate_user, create_access_token
from main import app
from rbac import can_access_documents, is_valid_role


def test_valid_login_returns_bearer_token():
    response = TestClient(app).post(
        "/login",
        json={"username": "Natasha", "password": "hrpass123"},
    )
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    assert response.json()["role"] == "hr"


def test_invalid_credentials_return_401():
    client = TestClient(app)
    assert client.post("/login", json={"username": "Natasha", "password": "bad"}).status_code == 401
    assert client.post("/login", json={"username": "unknown", "password": "bad"}).status_code == 401


def test_roles_are_centralized_and_scoped():
    assert is_valid_role("hr")
    assert is_valid_role("engineering")
    assert is_valid_role("c_level")
    assert not is_valid_role("marketing")
    assert can_access_documents("hr", "hr")
    assert not can_access_documents("engineering", "hr")
    assert can_access_documents("c_level", "hr")


def test_token_role_is_issued_from_user_store():
    user = authenticate_user("Tony", "password123")
    assert user == {"username": "Tony", "role": "engineering"}
    token = create_access_token(user["username"], user["role"])
    response = TestClient(app).get("/test", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["role"] == "engineering"
