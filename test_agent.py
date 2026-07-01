"""
test_agent.py — Automated tests for the SHL Assessment Recommender API

Run with:
    pytest test_agent.py -v

Uses FastAPI TestClient to test the app directly — no running server required.
"""

import pytest
from fastapi.testclient import TestClient
from main import app


# ============================================================================
# Test Client Fixture
# ============================================================================

@pytest.fixture
def client():
    """Synchronous test client for FastAPI (no need for a running server)."""
    with TestClient(app) as c:
        yield c


# ============================================================================
# Test 1: Health Check
# ============================================================================

def test_health_check(client):
    """GET /health should return {"status": "ok"} with HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}


# ============================================================================
# Test 2: The Clarify Test
# ============================================================================

def test_clarify_vague_prompt(client):
    """A vague prompt should trigger a clarifying question, NOT recommendations."""
    payload = {
        "messages": [
            {"role": "user", "content": "I need to assess some people"}
        ]
    }

    response = client.post("/chat", json=payload)
    assert response.status_code == 200

    data = response.json()

    assert "reply" in data
    assert len(data["reply"]) > 0, "Agent reply should not be empty"

    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) == 0, (
        f"Expected empty recommendations for vague query, got {len(data['recommendations'])} items"
    )

    assert data["end_of_conversation"] is False

    print(f"\n[PASS] Clarify Test")
    print(f"   Agent asked: {data['reply'][:120]}...")


# ============================================================================
# Test 3: The Refusal Test
# ============================================================================

def test_refusal_legal_advice(client):
    """Asking for legal advice should be refused."""
    payload = {
        "messages": [
            {"role": "user", "content": "Does this test guarantee HIPAA compliance? Can you give me legal advice on employment law?"}
        ]
    }

    response = client.post("/chat", json=payload)
    assert response.status_code == 200

    data = response.json()

    assert "reply" in data
    assert len(data["reply"]) > 0, "Agent reply should not be empty"

    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) == 0, (
        f"Expected empty recommendations for legal question, got {len(data['recommendations'])} items"
    )

    assert data["end_of_conversation"] is False

    print(f"\n[PASS] Refusal Test")
    print(f"   Agent replied: {data['reply'][:120]}...")


# ============================================================================
# Test 4: The Success Test
# ============================================================================

def test_success_detailed_prompt(client):
    """A detailed prompt should produce immediate recommendations."""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I need an advanced Core Java and Spring test for a "
                    "senior backend engineer with 5+ years of experience. "
                    "They should also know SQL and work in an Agile environment."
                )
            }
        ]
    }

    response = client.post("/chat", json=payload)
    assert response.status_code == 200

    data = response.json()

    assert "reply" in data
    assert len(data["reply"]) > 0, "Agent reply should not be empty"

    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) >= 2, (
        f"Expected at least 2 recommendations for detailed query, got {len(data['recommendations'])}"
    )
    assert len(data["recommendations"]) <= 10, (
        f"Expected at most 10 recommendations, got {len(data['recommendations'])}"
    )

    for i, rec in enumerate(data["recommendations"]):
        assert "name" in rec and len(rec["name"]) > 0, f"Recommendation {i} missing name"
        assert "url" in rec and len(rec["url"]) > 0, f"Recommendation {i} missing url"
        assert "test_type" in rec and len(rec["test_type"]) > 0, f"Recommendation {i} missing test_type"
        assert "shl.com" in rec["url"], (
            f"Recommendation {i} URL doesn't point to shl.com: {rec['url']}"
        )

    assert data["end_of_conversation"] is False

    print(f"\n[PASS] Success Test")
    print(f"   Agent replied: {data['reply'][:120]}...")
    print(f"   Recommendations ({len(data['recommendations'])}):")
    for rec in data["recommendations"]:
        print(f"     - {rec['name']} ({rec['test_type']})")


# ============================================================================
# Test 5: Schema Compliance
# ============================================================================

def test_schema_compliance(client):
    """Every response must have exactly the required fields with correct types."""
    payload = {
        "messages": [
            {"role": "user", "content": "Hello, can you help me?"}
        ]
    }

    response = client.post("/chat", json=payload)
    assert response.status_code == 200

    data = response.json()

    required_keys = {"reply", "recommendations", "end_of_conversation"}
    assert required_keys.issubset(set(data.keys())), (
        f"Missing required keys. Got: {set(data.keys())}, need: {required_keys}"
    )

    assert isinstance(data["reply"], str), "reply must be a string"
    assert isinstance(data["recommendations"], list), "recommendations must be a list"
    assert isinstance(data["end_of_conversation"], bool), "end_of_conversation must be a boolean"

    print(f"\n[PASS] Schema Compliance Test")


# ============================================================================
# Test 6: Invalid Request (last message not from user)
# ============================================================================

def test_invalid_last_message_role(client):
    """If the last message is not from the user, return HTTP 400."""
    payload = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    }

    response = client.post("/chat", json=payload)
    assert response.status_code == 400

    print(f"\n[PASS] Invalid Request Test")
