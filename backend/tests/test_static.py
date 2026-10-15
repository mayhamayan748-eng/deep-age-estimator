"""Single-container mode: the API also serves the built frontend (STATIC_DIR)."""
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture(scope="module")
def static_client(models_dir, tmp_path_factory):
    dist = tmp_path_factory.mktemp("dist")
    (dist / "assets").mkdir()
    (dist / "index.html").write_text("<html>app</html>")
    (dist / "favicon.svg").write_text("<svg/>")
    (dist / "assets" / "index.js").write_text("console.log(1)")
    secret = dist.parent / "secret.txt"
    secret.write_text("do not serve")
    with TestClient(create_app(Settings(models_dir=models_dir, static_dir=dist))) as c:
        yield c


def test_serves_index_and_files(static_client):
    assert static_client.get("/").text == "<html>app</html>"
    assert static_client.get("/favicon.svg").text == "<svg/>"
    assert static_client.get("/assets/index.js").text == "console.log(1)"


def test_unknown_routes_fall_back_to_index(static_client):
    assert static_client.get("/some/client/route").text == "<html>app</html>"


def test_api_still_works_and_unknown_api_is_404(static_client):
    assert static_client.get("/api/health").json()["status"] == "ok"
    assert static_client.get("/api/nope").status_code == 404


def test_no_path_traversal(static_client):
    assert static_client.get("/..%2Fsecret.txt").text == "<html>app</html>"
