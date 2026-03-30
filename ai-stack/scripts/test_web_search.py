from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def test_web_search_endpoint_structure() -> None:
    from server import create_app
    from fastapi.testclient import TestClient
    
    app = create_app()
    client = TestClient(app)
    
    response = client.get("/web/search", params={"q": "test", "max_results": 3})
    
    _assert(response.status_code == 200, f"expected 200, got {response.status_code}")
    data = response.json()
    _assert("query" in data, "response should contain query")
    _assert("results" in data, "response should contain results")
    _assert("count" in data, "response should contain count")
    _assert(data["query"] == "test", "query should match")
    
    for result in data.get("results", []):
        _assert("title" in result, "result should have title")
        _assert("url" in result, "result should have url")
        _assert("snippet" in result, "result should have snippet")
    
    print("WEB_SEARCH_ENDPOINT_OK")


def test_web_search_budget_exceeded() -> None:
    from server import create_app
    from fastapi.testclient import TestClient
    
    app = create_app()
    client = TestClient(app)
    
    for i in range(10):
        response = client.get("/web/search", params={"q": f"test {i}"})
        _assert(response.status_code == 200, f"request {i+1} should succeed")
    
    response = client.get("/web/search", params={"q": "budget test"})
    _assert(response.status_code == 429, f"expected 429 on budget exceed, got {response.status_code}")
    
    print("WEB_SEARCH_BUDGET_OK")


def main() -> None:
    test_web_search_endpoint_structure()
    test_web_search_budget_exceeded()
    print("ALL_WEB_SEARCH_TESTS_OK")


if __name__ == "__main__":
    main()
