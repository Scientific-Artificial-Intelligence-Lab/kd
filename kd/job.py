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
KD_GPU_PROJECT_ID = int(os.environ.get("KD_GPU_PROJECT_ID", "1077376"))
JOB_OUTPUT_DIR = os.environ.get("KD_JOB_OUTPUT_DIR", "/home/outputs")

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
        save_dir = os.path.join(JOB_OUTPUT_DIR, f"kd_{job_id}")
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
    if "appAccessKey" not in sc:
        raise ValueError("Missing 'appAccessKey' cookie — not on Bohrium or session expired")
    if "clientName" not in sc:
        raise ValueError("Missing 'clientName' cookie — session expired")
    return sc["appAccessKey"].value, sc["clientName"].value


# ── Bohrium public implementations ────────────────────────────

def _bohrium_submit(model_name, dataset_name, params, request,
                    project_id=None, dataset_file=None):
    access_key, app_key = get_credentials(request)
    client = _client(access_key, app_key)

    inputs = {
        "model": model_name,
        "dataset_name": dataset_name or "burgers",
        "output_dir": "./outputs",
    }
    if params:
        for key, value in params.items():
            inputs[key] = value

    pid = project_id or KD_GPU_PROJECT_ID
    if not pid:
        raise RuntimeError(
            "No Bohrium project ID configured. "
            "Set the KD_GPU_PROJECT_ID environment variable."
        )

    res = client.app.job.submit(
        app_key=KD_GPU_APP_KEY,
        sub_model_name=KD_GPU_SUB_MODEL,
        project_id=pid,
        inputs=inputs,
        ext_config={"jobConfig": {"jobRead": 1}},
    )
    if isinstance(res, dict):
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
    res = client.app.job.list(page=1, page_size=5)
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
    info = client.app.job.detail(job_id)
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
        save_dir = os.path.join(JOB_OUTPUT_DIR, f"kd_{job_id}")
    os.makedirs(save_dir, exist_ok=True)

    client.app.job.download(
        job_id, remote_target="outputs", save_path=save_dir
    )

    result_path = os.path.join(save_dir, "result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f), save_dir

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
