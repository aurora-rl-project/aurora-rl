import asyncio
import hashlib
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
    app.state.stage_uploads = {}
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


def test_client_stage_chunk_uploads_delta_file(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    delta_file = delta_dir / "delta.safetensors"
    delta_file.write_bytes(b"chunked-delta-upload")
    app = make_app(tmp_path)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await stage_weights(
                [client],
                delta_dir,
                version="1",
                mode="delta",
                base_version="0",
                upload=True,
                upload_method="chunked",
                chunk_size_bytes=5,
            )

    results = asyncio.run(run())

    staged = app.state.staged_versions["1"]
    staged_path = staged["path"]
    assert [result.endpoint for result in results] == ["http://test"]
    assert [result.operation for result in results] == ["stage_weights"]
    assert all(result.ok for result in results)
    assert app.state.stage_uploads == {}
    assert staged["owned"] is True
    assert staged["mode"] == "delta"
    assert staged_path.parent == app.state.staging_dir
    assert staged_path.read_bytes() == b"chunked-delta-upload"


def test_client_stage_chunk_uploads_delta_file_to_multiple_endpoints(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    delta_file = delta_dir / "delta.safetensors"
    delta_file.write_bytes(b"multi-endpoint-delta")
    app_dirs = [tmp_path / "a", tmp_path / "b"]
    for app_dir in app_dirs:
        app_dir.mkdir()
    apps = [make_app(app_dir) for app_dir in app_dirs]

    async def run() -> None:
        clients = [
            httpx.AsyncClient(transport=httpx.ASGITransport(app=apps[0]), base_url="http://worker-a"),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=apps[1]), base_url="http://worker-b"),
        ]
        async with clients[0], clients[1]:
            return await stage_weights(
                clients,
                delta_dir,
                version="1",
                mode="delta",
                base_version="0",
                upload=True,
                upload_method="chunked",
                chunk_size_bytes=4,
            )

    results = asyncio.run(run())

    assert sorted(result.endpoint for result in results) == ["http://worker-a", "http://worker-b"]
    assert all(result.ok for result in results)
    for app in apps:
        staged = app.state.staged_versions["1"]
        assert staged["owned"] is True
        assert staged["path"].read_bytes() == b"multi-endpoint-delta"


def test_client_stage_streams_growing_delta_file(tmp_path) -> None:
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    delta_file = delta_dir / "delta.safetensors"
    stable_file = delta_dir / "STABLE"
    app = make_app(tmp_path)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            stage_task = asyncio.create_task(
                stage_weights(
                    [client],
                    delta_dir,
                    version="1",
                    mode="delta",
                    base_version="0",
                    upload=True,
                    upload_method="streaming",
                    chunk_size_bytes=4,
                    done_path=stable_file,
                    poll_interval_s=0.01,
                )
            )

            await asyncio.sleep(0.02)
            delta_file.write_bytes(b"stream-")
            await asyncio.sleep(0.02)
            with delta_file.open("ab") as f:
                f.write(b"delta-upload")
            stable_file.touch()
            return await stage_task

    results = asyncio.run(run())

    staged = app.state.staged_versions["1"]
    assert [result.endpoint for result in results] == ["http://test"]
    assert all(result.ok for result in results)
    assert staged["owned"] is True
    assert staged["mode"] == "delta"
    assert staged["path"].read_bytes() == b"stream-delta-upload"


def test_stage_stream_finalize_accepts_complete_upload(tmp_path) -> None:
    app = make_app(tmp_path)
    content = b"streaming-delta"
    digest = hashlib.sha256(content).hexdigest()

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            init_response = await client.post(
                "/stage_stream_init",
                json={"version": "1", "mode": "delta", "base_version": "0", "filename": "delta.safetensors"},
            )
            assert init_response.status_code == 200
            upload_id = init_response.json()["upload_id"]

            first_response = await client.post(
                "/stage_stream_chunk",
                params={"upload_id": upload_id, "offset": "0"},
                content=content[:6],
            )
            assert first_response.status_code == 200
            second_response = await client.post(
                "/stage_stream_chunk",
                params={"upload_id": upload_id, "offset": "6"},
                content=content[6:],
            )
            assert second_response.status_code == 200

            finalize_response = await client.post(
                "/stage_stream_finalize",
                data={"upload_id": upload_id, "final_size": str(len(content)), "sha256": digest},
            )
            assert finalize_response.status_code == 200

    asyncio.run(run())

    staged = app.state.staged_versions["1"]
    assert staged["owned"] is True
    assert staged["path"].read_bytes() == content
    assert app.state.stage_uploads == {}


def test_stage_stream_finalize_rejects_missing_range(tmp_path) -> None:
    app = make_app(tmp_path)
    content = b"incomplete"

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            init_response = await client.post(
                "/stage_stream_init",
                json={"version": "1", "mode": "delta", "base_version": "0", "filename": "delta.safetensors"},
            )
            assert init_response.status_code == 200
            upload_id = init_response.json()["upload_id"]

            chunk_response = await client.post(
                "/stage_stream_chunk",
                params={"upload_id": upload_id, "offset": "0"},
                content=content[:5],
            )
            assert chunk_response.status_code == 200
            finalize_response = await client.post(
                "/stage_stream_finalize",
                data={"upload_id": upload_id, "final_size": str(len(content))},
            )
            assert finalize_response.status_code == 409

    asyncio.run(run())

    assert app.state.staged_versions == {}
    assert app.state.stage_uploads


def test_stage_stream_finalize_rejects_hash_mismatch(tmp_path) -> None:
    app = make_app(tmp_path)
    content = b"corrupted"

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            init_response = await client.post(
                "/stage_stream_init",
                json={"version": "1", "mode": "delta", "base_version": "0", "filename": "delta.safetensors"},
            )
            assert init_response.status_code == 200
            upload_id = init_response.json()["upload_id"]

            chunk_response = await client.post(
                "/stage_stream_chunk",
                params={"upload_id": upload_id, "offset": "0"},
                content=content,
            )
            assert chunk_response.status_code == 200
            finalize_response = await client.post(
                "/stage_stream_finalize",
                data={"upload_id": upload_id, "final_size": str(len(content)), "sha256": "0" * 64},
            )
            assert finalize_response.status_code == 409

    asyncio.run(run())

    assert app.state.staged_versions == {}
    assert app.state.stage_uploads == {}


def test_stage_chunk_finalize_accepts_out_of_order_chunks(tmp_path) -> None:
    app = make_app(tmp_path)
    content = b"first-second"
    digest = hashlib.sha256(content).hexdigest()

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            common_data = {
                "version": "1",
                "mode": "delta",
                "base_version": "0",
                "filename": "delta.safetensors",
                "total_size": str(len(content)),
                "sha256": digest,
            }
            second_response = await client.post(
                "/stage_chunk",
                data={**common_data, "offset": "6"},
                files={"file": ("delta.safetensors", content[6:], "application/octet-stream")},
            )
            assert second_response.status_code == 200
            first_response = await client.post(
                "/stage_chunk",
                data={**common_data, "offset": "0"},
                files={"file": ("delta.safetensors", content[:6], "application/octet-stream")},
            )
            assert first_response.status_code == 200
            finalize_response = await client.post("/stage_finalize", data=common_data)
            assert finalize_response.status_code == 200

    asyncio.run(run())

    staged_path = app.state.staged_versions["1"]["path"]
    assert staged_path.read_bytes() == content


def test_stage_chunk_finalize_rejects_missing_chunk(tmp_path) -> None:
    app = make_app(tmp_path)
    content = b"incomplete"
    digest = hashlib.sha256(content).hexdigest()

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            common_data = {
                "version": "1",
                "mode": "delta",
                "base_version": "0",
                "filename": "delta.safetensors",
                "total_size": str(len(content)),
                "sha256": digest,
            }
            chunk_response = await client.post(
                "/stage_chunk",
                data={**common_data, "offset": "0"},
                files={"file": ("delta.safetensors", content[:5], "application/octet-stream")},
            )
            assert chunk_response.status_code == 200
            finalize_response = await client.post("/stage_finalize", data=common_data)
            assert finalize_response.status_code == 409

    asyncio.run(run())

    assert app.state.staged_versions == {}
    assert app.state.stage_uploads


def test_stage_chunk_finalize_rejects_hash_mismatch(tmp_path) -> None:
    app = make_app(tmp_path)
    content = b"corrupted"

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            common_data = {
                "version": "1",
                "mode": "delta",
                "base_version": "0",
                "filename": "delta.safetensors",
                "total_size": str(len(content)),
                "sha256": "0" * 64,
            }
            chunk_response = await client.post(
                "/stage_chunk",
                data={**common_data, "offset": "0"},
                files={"file": ("delta.safetensors", content, "application/octet-stream")},
            )
            assert chunk_response.status_code == 200
            finalize_response = await client.post("/stage_finalize", data=common_data)
            assert finalize_response.status_code == 409

    asyncio.run(run())

    assert app.state.staged_versions == {}
    assert app.state.stage_uploads == {}


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
