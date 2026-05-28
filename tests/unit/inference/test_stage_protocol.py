import asyncio
from urllib.parse import parse_qs

import httpx
from fastapi import FastAPI

from prime_rl.inference.vllm.server import router
from prime_rl.utils.client import commit_weights, reload_weights, stage_weights


class FakeEngineClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    async def collective_rpc(self, method: str, args: tuple = ()) -> None:
        self.calls.append((method, args))


def make_app(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = FakeEngineClient()
    app.state.staging_dir = tmp_path / "staging"
    app.state.staging_dir.mkdir()
    app.state.staged_versions = {}
    app.state.active_version = None
    return app


def test_stage_commit_delta_path_protocol(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    (delta_dir / "delta.safetensors").write_bytes(b"placeholder")
    app = make_app(tmp_path)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            stage_response = await client.post(
                "/stage",
                data={"version": "1", "mode": "delta", "path": delta_dir.as_posix()},
            )
            assert stage_response.status_code == 200

            commit_response = await client.post("/commit", data={"version": "1", "mode": "delta"})
            assert commit_response.status_code == 200

    asyncio.run(run())

    assert app.state.active_version == "1"
    assert app.state.engine_client.calls == [("update_weights_from_delta_path", (delta_dir.as_posix(),))]


def test_stage_delta_rejects_base_version_mismatch(tmp_path) -> None:
    delta_dir = tmp_path / "step_2"
    delta_dir.mkdir()
    (delta_dir / "delta.safetensors").write_bytes(b"placeholder")
    app = make_app(tmp_path)
    app.state.active_version = "1"

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            mismatch_response = await client.post(
                "/stage",
                data={"version": "2", "mode": "delta", "base_version": "0", "path": delta_dir.as_posix()},
            )
            assert mismatch_response.status_code == 409
            assert app.state.staged_versions == {}

            stage_response = await client.post(
                "/stage",
                data={"version": "2", "mode": "delta", "base_version": "1", "path": delta_dir.as_posix()},
            )
            assert stage_response.status_code == 200

    asyncio.run(run())

    assert app.state.staged_versions["2"]["path"] == delta_dir


def test_reload_weights_clears_staging_state(tmp_path) -> None:
    app = make_app(tmp_path)
    staged_file = app.state.staging_dir / "1_delta_delta.safetensors"
    staged_file.write_bytes(b"placeholder")
    app.state.staged_versions = {"1": {"path": staged_file, "mode": "delta", "owned": True}}
    app.state.active_version = "1"

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/reload_weights")
            assert response.status_code == 200

    asyncio.run(run())

    assert app.state.active_version is None
    assert app.state.staged_versions == {}
    assert not staged_file.exists()
    assert app.state.engine_client.calls == [("reload_weights", ())]


def test_client_stage_uploads_delta_file(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    delta_file = delta_dir / "delta.safetensors"
    delta_file.write_bytes(b"uploaded-delta")
    app = make_app(tmp_path)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await stage_weights([client], delta_dir, version="1", mode="delta", base_version="0", upload=True)

    asyncio.run(run())

    staged = app.state.staged_versions["1"]
    staged_path = staged["path"]
    assert staged["owned"] is True
    assert staged["mode"] == "delta"
    assert staged_path.parent == app.state.staging_dir
    assert staged_path.read_bytes() == b"uploaded-delta"


def test_client_stage_commit_reload_helpers(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/stage":
            form = parse_qs(request.content.decode())
            assert form["version"] == ["1"]
            assert form["mode"] == ["delta"]
            assert form["path"] == [delta_dir.as_posix()]
        if request.url.path == "/commit":
            form = parse_qs(request.content.decode())
            assert form["version"] == ["1"]
            assert form["mode"] == ["delta"]
        return httpx.Response(200, json={"status": "ok"})

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await stage_weights([client], delta_dir, version="1", mode="delta")
            await commit_weights([client], version="1", mode="delta")
            await reload_weights([client])

    asyncio.run(run())

    assert [request.url.path for request in requests] == [
        "/stage",
        "/pause",
        "/commit",
        "/resume",
        "/pause",
        "/reload_weights",
        "/resume",
    ]
