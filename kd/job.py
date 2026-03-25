"""Bohrium offline task helpers for GPU job submission and monitoring.

Used by ``app.py`` (running on a CPU web node) to submit GPU-intensive
models as Bohrium offline tasks, poll their status, and download results.

The GPU offline app uses the Launching framework (dp.launching).
Inputs are passed as flat key-value pairs matching the Options schema
defined in ``launching.py``.

Env vars (set in the Bohrium app image or .env):
    KD_GPU_APP_KEY      — app_key of the registered GPU offline app
    KD_GPU_SUB_MODEL    — sub_model_name for the offline app
    KD_GPU_PROJECT_ID   — default Bohrium project ID (0 = auto-detect)
    KD_JOB_OUTPUT_DIR   — local dir for downloading job outputs
    KD_JOB_BACKEND      — "bohrium" (default) or "mock" (local JSON files)
"""

import json
import os
import time

# ── Configuration ─────────────────────────────────────────────

KD_GPU_APP_KEY = os.environ.get("KD_GPU_APP_KEY", "kd-gpu")
KD_GPU_SUB_MODEL = os.environ.get("KD_GPU_SUB_MODEL", "Options")
KD_GPU_PROJECT_ID = int(os.environ.get("KD_GPU_PROJECT_ID", "0"))
JOB_OUTPUT_DIR = os.environ.get("KD_JOB_OUTPUT_DIR", "/data/outputs")

KD_JOB_BACKEND = os.environ.get("KD_JOB_BACKEND", "bohrium")
MOCK_DIR = "/tmp/kd_mock_jobs"

JOB_STATUS_MAP = {
    0: "Pending", 1: "Running", 2: "Packaging", 3: "Submitted",
    4: "Running", 5: "Running", 6: "Running", 7: "Running",
    8: "Running", 9: "DownloadOutput", 10: "Finished",
    11: "Finished", 12: "Failed", 13: "Error",
}


# ── Mock backend ─────────────────────────────────────────────
# When KD_JOB_BACKEND=mock, jobs are simulated with local JSON files
# under MOCK_DIR (/tmp/kd_mock_jobs).  Each job is stored as
# <MOCK_DIR>/<job_id>.json with metadata including a created_at
# timestamp.  Status auto-advances: Pending (0-2s) → Running (2-8s)
# → Finished (>8s).

_MOCK_PENDING_SECS = 2
_MOCK_RUNNING_SECS = 8


def _mock_dir():
    os.makedirs(MOCK_DIR, exist_ok=True)
    return MOCK_DIR


def _mock_job_path(job_id):
    return os.path.join(_mock_dir(), f"{job_id}.json")


def _mock_next_id():
    """Allocate a sequential mock job ID."""
    counter_path = os.path.join(_mock_dir(), "_counter.json")
    if os.path.exists(counter_path):
        with open(counter_path) as f:
            ctr = json.load(f)["next"]
    else:
        ctr = 90000
    with open(counter_path, "w") as f:
        json.dump({"next": ctr + 1}, f)
    return ctr


def _mock_status_code(meta):
    """Derive status code from elapsed time since job creation."""
    elapsed = time.time() - meta["created_at"]
    if elapsed < _MOCK_PENDING_SECS:
        return 0   # Pending
    if elapsed < _MOCK_RUNNING_SECS:
        return 4   # Running
    return 10      # Finished


def _mock_submit(model_name, dataset_name, params, request,
                 project_id=None, dataset_file=None):
    job_id = _mock_next_id()
    meta = {
        "job_id": job_id,
        "model": model_name,
        "dataset": dataset_name,
        "params": params or {},
        "created_at": time.time(),
    }
    with open(_mock_job_path(job_id), "w") as f:
        json.dump(meta, f, indent=2)
    return job_id


def _mock_check(job_id, request):
    path = _mock_job_path(job_id)
    if not os.path.exists(path):
        return {
            "status_code": -1,
            "status_text": "NotFound",
            "finished": False,
            "failed": True,
            "spend_time": "",
        }
    with open(path) as f:
        meta = json.load(f)
    code = _mock_status_code(meta)
    return {
        "status_code": code,
        "status_text": JOB_STATUS_MAP.get(code, f"Unknown({code})"),
        "finished": code in (10, 11),
        "failed": code in (12, 13),
        "spend_time": f"{time.time() - meta['created_at']:.1f}s",
    }


def _mock_download(job_id, request, save_dir=None):
    path = _mock_job_path(job_id)
    if not os.path.exists(path):
        return {"status": "error", "equation": "(mock job not found)"}, ""
    with open(path) as f:
        meta = json.load(f)

    if save_dir is None:
        save_dir = os.path.join(JOB_OUTPUT_DIR, f"{job_id}")
    os.makedirs(save_dir, exist_ok=True)

    result = {
        "status": "ok",
        "equation": "u = mock_result(x, t)",
        "model": meta.get("model", ""),
        "dataset": meta.get("dataset", ""),
        "mock": True,
    }
    result_path = os.path.join(save_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return result, save_dir


def _mock_find_active(request):
    """Return the most recent non-finished mock job, or None."""
    d = _mock_dir()
    best_id, best_time = None, 0
    for fname in os.listdir(d):
        if fname.startswith("_") or not fname.endswith(".json"):
            continue
        with open(os.path.join(d, fname)) as f:
            meta = json.load(f)
        code = _mock_status_code(meta)
        if 0 <= code <= 9 and meta["created_at"] > best_time:
            best_id = meta["job_id"]
            best_time = meta["created_at"]
    return best_id


# ── Bohrium internal helpers ─────────────────────────────────

def _client(access_key, app_key):
    from bohrium_open_sdk import OpenSDK
    return OpenSDK(access_key=access_key, app_key=app_key)


def get_credentials(request):
    """Extract Bohrium credentials from Gradio request cookies.

    On the Bohrium platform the cookies ``appAccessKey`` and ``clientName``
    are injected automatically by the gateway.
    """
    from http.cookies import SimpleCookie
    if request is None:
        raise ValueError("No request object — cannot extract Bohrium credentials")
    cookie_str = request.headers.get("cookie", "")
    if not cookie_str:
        raise ValueError("No cookies in request — session may have expired")
    sc = SimpleCookie()
    sc.load(cookie_str)
    print(f"[kd] All cookie keys: {list(sc.keys())}", flush=True)
    if "appAccessKey" not in sc:
        raise ValueError("Missing 'appAccessKey' cookie — not on Bohrium or session expired")
    if "clientName" not in sc:
        raise ValueError("Missing 'clientName' cookie — session expired")
    return sc["appAccessKey"].value, sc["clientName"].value


# ── Bohrium public implementations ────────────────────────────

def _get_user_project_id(client):
    """Get the user's first available project ID from Bohrium.

    Try SDK method first, then fall back to REST endpoint.
    """
    import json

    # Strategy 1: try SDK method (in case it works in deployed env)
    try:
        if hasattr(client.user, "list_project"):
            res = client.user.list_project()
            print(f"[kd] SDK list_project response: {res}", flush=True)
            data = res.get("data", res) if isinstance(res, dict) else res
            items = data.get("items", []) if isinstance(data, dict) else []
            if items:
                pid = items[0].get("id") or items[0].get("project_id") or items[0].get("projectId")
                if pid:
                    print(f"[kd] Got project_id={pid} from SDK list_project", flush=True)
                    return int(pid)
        else:
            print("[kd] SDK has no list_project method", flush=True)
    except Exception as e:
        print(f"[kd] SDK list_project failed: {e}", flush=True)

    # Strategy 2: REST endpoint (confirmed working)
    try:
        r = client.get("openapi/v1/project/list", params={"page": 1, "pageSize": 5})
        print(f"[kd] REST project/list: status={r.status_code} body={r.text[:500]}", flush=True)
        if r.status_code == 200:
            res = json.loads(r.text)
            items = res.get("data", {}).get("items", [])
            if items:
                pid = items[0].get("id")
                print(f"[kd] Got user project_id={pid} from REST", flush=True)
                return int(pid)
    except Exception as e:
        print(f"[kd] REST project/list failed: {e}", flush=True)

    print("[kd] All strategies to get project_id failed", flush=True)
    return None


def _bohrium_submit(model_name, dataset_name, params, request,
                    project_id=None, dataset_file=None):
    access_key, app_key = get_credentials(request)
    client = _client(access_key, app_key)

    inputs = {
        "model": model_name,
        "dataset_name": dataset_name or "burgers",
        "output_dir": "output",
    }
    if params:
        for key, value in params.items():
            inputs[key] = value

    # Resolve project_id: explicit arg > env var > try user API
    pid = project_id if project_id is not None else KD_GPU_PROJECT_ID
    if pid == 0:
        user_pid = _get_user_project_id(client)
        if user_pid:
            pid = user_pid
        else:
            raise RuntimeError(
                "Cannot determine Bohrium project ID. "
                "Please ensure you have at least one project on Bohrium."
            )

    res = client.app.job.submit(
        app_key=KD_GPU_APP_KEY,
        sub_model_name=KD_GPU_SUB_MODEL,
        project_id=pid,
        inputs=inputs,
        ext_config={"jobConfig": {"jobRead": 1}},
    )
    if isinstance(res, dict):
        # API error response
        if "error" in res and "code" in res:
            msg = res["error"].get("msg", str(res["error"]))
            raise RuntimeError(f"Bohrium API error: {msg}")
        if "data" in res:
            return res["data"]["jobId"]
        if "jobId" in res:
            return res["jobId"]
        if "job_id" in res:
            return res["job_id"]
    raise RuntimeError(f"Unexpected submit response: {res}")


def _bohrium_find_active(request):
    access_key, app_key = get_credentials(request)
    client = _client(access_key, app_key)
    try:
        res = client.app.job.list(page=1, page_size=5)
    except SystemExit as e:
        # SDK calls exit(1) on 401 — catch to prevent process termination
        print(f"[kd] _bohrium_find_active: SDK called exit({e.code}), returning None", flush=True)
        return None
    except Exception as e:
        print(f"[kd] _bohrium_find_active: job.list failed: {e}", flush=True)
        return None
    data = res.get("data", res) if isinstance(res, dict) else res
    items = data.get("items", []) if isinstance(data, dict) else []
    for item in items:
        status = item.get("status", -1)
        if 0 <= status <= 9:
            return item.get("jobId") or item.get("job_id")
    return None


def _bohrium_check(job_id, request):
    access_key, app_key = get_credentials(request)
    client = _client(access_key, app_key)
    try:
        info = client.app.job.detail(job_id)
    except SystemExit as e:
        print(f"[kd] _bohrium_check: SDK called exit({e.code}), treating as error", flush=True)
        raise RuntimeError(f"Bohrium SDK exited with code {e.code}") from e
    data = info.get("data", info) if isinstance(info, dict) else info
    code = data.get("status", -1) if isinstance(data, dict) else -1
    return {
        "status_code": code,
        "status_text": JOB_STATUS_MAP.get(code, f"Unknown({code})"),
        "finished": code in (10, 11),
        "failed": code in (12, 13),
        "spend_time": data.get("spendTime", "") if isinstance(data, dict) else "",
    }


def _bohrium_download(job_id, request, save_dir=None):
    access_key, app_key = get_credentials(request)
    client = _client(access_key, app_key)

    if save_dir is None:
        save_dir = os.path.join(JOB_OUTPUT_DIR, f"{job_id}")
    os.makedirs(save_dir, exist_ok=True)

    # Remove stale result.json to prevent SDK append-corruption
    result_path = os.path.join(save_dir, "result.json")
    if os.path.exists(result_path):
        os.remove(result_path)

    # Try official SDK method first (web.sub_model.download),
    # fall back to app.job.download if not available.
    downloaded = False
    if hasattr(client, "web") and hasattr(client.web, "sub_model"):
        try:
            print(f"[kd] Trying web.sub_model.download(job_id={job_id})", flush=True)
            client.web.sub_model.download(
                job_id, remote_target="output", save_path=save_dir
            )
            downloaded = True
            print(f"[kd] web.sub_model.download OK", flush=True)
        except Exception as e:
            print(f"[kd] web.sub_model.download failed: {e}, trying app.job.download", flush=True)

    if not downloaded:
        print(f"[kd] Using app.job.download(job_id={job_id})", flush=True)
        try:
            client.app.job.download(
                job_id, remote_target="output", save_path=save_dir
            )
        except SystemExit as e:
            print(f"[kd] app.job.download: SDK called exit({e.code})", flush=True)
            raise RuntimeError(f"Bohrium SDK exited with code {e.code} during download") from e
        print(f"[kd] app.job.download OK", flush=True)

    print(f"[kd] Download dir contents: {os.listdir(save_dir)}", flush=True)
    s_dir = os.path.join(save_dir, "s")
    if os.path.isdir(s_dir):
        for root, dirs, files in os.walk(s_dir):
            for fname in files:
                print(f"[kd]   s/{os.path.relpath(os.path.join(root, fname), s_dir)}", flush=True)

    # Bohrium SDK downloads files into an 's/' subdirectory.
    # Search result.json in: save_dir, save_dir/s/, save_dir/s/output/
    result_path = None
    for candidate in [
        os.path.join(save_dir, "result.json"),
        os.path.join(save_dir, "s", "result.json"),
        os.path.join(save_dir, "s", "output", "result.json"),
        os.path.join(save_dir, "s", "outputs", "result.json"),
    ]:
        if os.path.exists(candidate):
            result_path = candidate
            break

    print(f"[kd] result.json found at: {result_path}", flush=True)

    if result_path:
        with open(result_path) as f:
            raw = f.read()
        print(f"[kd] result.json size={len(raw)} bytes", flush=True)
        try:
            return json.loads(raw), save_dir
        except json.JSONDecodeError:
            # Fallback: raw_decode handles "Extra data" from concat'd JSON
            try:
                obj, _ = json.JSONDecoder().raw_decode(raw)
                return obj, save_dir
            except json.JSONDecodeError:
                return {"status": "error", "equation": "(result.json parse error)"}, save_dir

    print(f"[kd] result.json not found in {save_dir}", flush=True)
    return {"status": "error", "equation": "(result.json not found)"}, save_dir


# ── Public API (routing layer) ───────────────────────────────

def _is_mock():
    return KD_JOB_BACKEND == "mock"


def submit_gpu_job(model_name, dataset_name, params, request,
                   project_id=None, dataset_file=None):
    """Submit a GPU training job. Backend selected by KD_JOB_BACKEND."""
    if _is_mock():
        return _mock_submit(model_name, dataset_name, params, request,
                            project_id, dataset_file)
    return _bohrium_submit(model_name, dataset_name, params, request,
                           project_id, dataset_file)


def find_active_job(request):
    """Find the most recent running/pending job."""
    if _is_mock():
        return _mock_find_active(request)
    return _bohrium_find_active(request)


def check_job(job_id, request):
    """Check the status of a submitted GPU job."""
    if _is_mock():
        return _mock_check(job_id, request)
    return _bohrium_check(job_id, request)


def download_result(job_id, request, save_dir=None):
    """Download completed job output and parse result.json."""
    if _is_mock():
        return _mock_download(job_id, request, save_dir)
    return _bohrium_download(job_id, request, save_dir)
