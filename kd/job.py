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
"""

import json
import os

# ── Configuration ─────────────────────────────────────────────

KD_GPU_APP_KEY = os.environ.get("KD_GPU_APP_KEY", "kd-gpu")
KD_GPU_SUB_MODEL = os.environ.get("KD_GPU_SUB_MODEL", "Options")
KD_GPU_PROJECT_ID = int(os.environ.get("KD_GPU_PROJECT_ID", "1077376"))
JOB_OUTPUT_DIR = os.environ.get("KD_JOB_OUTPUT_DIR", "/home/outputs")

JOB_STATUS_MAP = {
    0: "Pending", 1: "Running", 2: "Packaging", 3: "Submitted",
    4: "Running", 5: "Running", 6: "Running", 7: "Running",
    8: "Running", 9: "DownloadOutput", 10: "Finished",
    11: "Finished", 12: "Failed", 13: "Error",
}


# ── Internal helpers ──────────────────────────────────────────

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


# ── Public API ────────────────────────────────────────────────

def submit_gpu_job(model_name, dataset_name, params, request,
                   project_id=None, dataset_file=None):
    """Submit a GPU training job to the Bohrium offline task platform.

    Inputs are passed as flat key-value pairs matching the Options schema
    in ``launching.py`` (Launching framework).

    Args:
        model_name: Runner model key (``"dlga"`` or ``"dscv_spr"``).
        dataset_name: Built-in dataset name (e.g. ``"burgers"``).
        params: Model hyper-parameter dict matching launching.py Options.
        request: ``gr.Request`` object (for auth cookies).
        project_id: Override Bohrium project ID.
        dataset_file: Optional local path to a user-uploaded data file.

    Returns:
        int: The submitted job ID.
    """
    access_key, app_key = get_credentials(request)
    client = _client(access_key, app_key)

    # Build flat inputs matching launching.py Options schema
    inputs = {
        "model": model_name,
        "dataset_name": dataset_name or "burgers",
        "output_dir": "./outputs",
    }
    # Merge model-specific params directly into inputs
    if params:
        for key, value in params.items():
            inputs[key] = value

    # Resolve project ID
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
    # Handle different response formats
    if isinstance(res, dict):
        if "data" in res:
            return res["data"]["jobId"]
        if "jobId" in res:
            return res["jobId"]
        if "job_id" in res:
            return res["job_id"]
    raise RuntimeError(f"Unexpected submit response: {res}")


def check_job(job_id, request):
    """Check the status of a submitted GPU job.

    Returns:
        dict with keys:
            status_code (int), status_text (str),
            finished (bool), failed (bool), spend_time (str).
    """
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


def download_result(job_id, request, save_dir=None):
    """Download completed job output and parse ``result.json``.

    Returns:
        tuple: ``(result_dict, save_dir_path)``
    """
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
