"""
Tests for CORS configuration compliance:
- Verified trusted origins receive access-control-allow-origin and credentials
- Untrusted origins are denied
- Wildcard '*' combined with allow_credentials=True is prevented
"""

from fastapi.testclient import TestClient
from api import app, DEFAULT_TRUSTED_ORIGINS


def test_cors_trusted_origin_allowed():
    client = TestClient(app)
    trusted_origin = "http://localhost:8501"

    response = client.options(
        "/predict/win-probability",
        headers={
            "Origin": trusted_origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"
        }
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == trusted_origin
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_untrusted_origin_rejected():
    client = TestClient(app)
    untrusted_origin = "http://malicious-website.com"

    response = client.options(
        "/predict/win-probability",
        headers={
            "Origin": untrusted_origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"
        }
    )
    # Untrusted origins should NOT have access-control-allow-origin set to the untrusted origin
    assert response.headers.get("access-control-allow-origin") != untrusted_origin


def test_cors_middleware_does_not_use_wildcard_with_credentials():
    # Verify CORS configuration across app middleware
    for middleware in app.user_middleware:
        if "CORSMiddleware" in str(middleware.cls):
            kwargs = getattr(middleware, "kwargs", {})
            allow_origins = kwargs.get("allow_origins", [])
            allow_credentials = kwargs.get("allow_credentials", False)
            if allow_credentials:
                assert "*" not in allow_origins, "CORS violation: allow_origins contains '*' while allow_credentials is True"
                assert len(allow_origins) > 0, "Trusted origins must be explicitly specified"

