from fastapi.testclient import TestClient

import main


def test_health_is_public():
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_requires_authentication():
    response = TestClient(main.app).post("/chat", json={"message": "hello"})
    assert response.status_code == 401


def test_chat_uses_authenticated_user_and_clean_response(monkeypatch):
    monkeypatch.setattr(
        main.chatbot,
        "chatbot_service",
        lambda role, question: {"answer": "Engineering answer", "sources": ["engineering_master_doc.md"]},
    )
    client = TestClient(main.app)
    login = client.post("/login", json={"username": "Tony", "password": "password123"})
    token = login.json()["access_token"]
    response = client.post(
        "/chat",
        json={"message": "Tell me about engineering"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Engineering answer",
        "sources": ["engineering_master_doc.md"],
        "user": {"username": "Tony", "role": "engineering"},
    }


def test_chat_rejects_client_role():
    client = TestClient(main.app)
    login = client.post("/login", json={"username": "Tony", "password": "password123"})
    token = login.json()["access_token"]
    response = client.post(
        "/chat",
        json={"message": "hello", "role": "hr"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422
