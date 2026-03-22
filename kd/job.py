"""Bohrium offline task helpers for GPU job submission and monitoring.

Used by ``app.py`` (running on a CPU web node) to submit GPU-intensive
models as Bohrium offline tasks, poll their status, and download results.

Configuration is read from environment variables so the same ``app.py``
works both locally (no SDK, all models run in-process) and on Bohrium.

Env vars (set in the Bohrium app image or .env):
    KD_GPU_APP_KEY      — app_key of the registered GPU offline app
    KD_GPU_SUB_MODEL    — sub_model_name for the offline app
    KD_GPU_PROJECT_ID   — default Bohrium project ID (0 = auto-detect)
    KD_JOB_OUTPUT_DIR   — local dir for downloading job outputs
"""

import json
import os
import tempfile

# ── Configuration ─────────────────────────────────────────────

KD_GPU_APP_KEY = os.environ.get("KD_GPU_APP_KEY", "kd-gpu-runner")
KD_GPU_SUB_MODEL = os.environ.get("KD_GPU_SUB_MODEL", "KD_GPU")
KD_GPU_PROJECT_ID = int(os.environ.get("KD_GPU_PROJECT_ID", "0"))
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
    cookie_str = request.headers.get("cookie", "")
    sc = SimpleCookie()
    sc.load(cookie_str)
    return sc["appAccessKey"].value, sc["clientName"].value


# ── Public API ────────────────────────────────────────────────

def submit_gpu_job(model_name, dataset_name, params, request,
                   project_id=None, dataset_file=None):
    """Submit a GPU training job to the Bohrium offline task platform.

    Args:
        model_name: Runner model key (``"dlga"`` or ``"dscv_spr"``).
        dataset_name: Built-in dataset name (e.g. ``"burgers"``).
        params: Model hyper-parameter dict (forwarded to ``runner.py``).
        request: ``gr.Request`` object (for auth cookies).
        project_id: Override Bohrium project ID.
        dataset_file: Optional local path to a user-uploaded data file.

    Returns:
        int: The submitted job ID.
    """
    from bohrium_open_sdk import UploadInputItem

    access_key, app_key = get_credentials(request)
    client = _client(access_key, app_key)

    # Build the runner config JSON
    config = {
        "model": model_name,
        "output_dir": "output",
        "params": params,
    }
    if dataset_name:
        config["dataset_name"] = dataset_name

    # Write to a temp file for upload
    fd, config_path = tempfile.mkstemp(suffix=".json", prefix="kd_cfg_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, ensure_ascii=False)

        inputs = {"config": "", "output_dir": "output"}
        upload_files = [UploadInputItem(input_field="config", src=config_path)]

        if dataset_file and os.path.exists(dataset_file):
            inputs["dataset_file"] = ""
            upload_files.append(
                UploadInputItem(input_field="dataset_file", src=dataset_file)
            )

        # Resolve project ID
        pid = project_id or KD_GPU_PROJECT_ID
        if not pid:
            proj_res = client.user.list_project()
            items = proj_res.get("data", {}).get("items", [])
            if items:
                pid = items[0]["project_id"]
            else:
                raise RuntimeError(
                    "No Bohrium project found. "
                    "Set the KD_GPU_PROJECT_ID environment variable."
                )

        res = client.app.job.submit(
            app_key=KD_GPU_APP_KEY,
            sub_model_name=KD_GPU_SUB_MODEL,
            project_id=pid,
            inputs=inputs,
            upload_files=upload_files,
        )
        return res["data"]["jobId"]
    finally:
        try:
            os.unlink(config_path)
        except OSError:
            pass


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
    code = info["data"]["status"]
    return {
        "status_code": code,
        "status_text": JOB_STATUS_MAP.get(code, f"Unknown({code})"),
        "finished": code in (10, 11),
        "failed": code in (12, 13),
        "spend_time": info["data"].get("spendTime", ""),
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

    client.web.sub_model.download(
        job_id, remote_target="output", save_path=save_dir
    )

    result_path = os.path.join(save_dir, "result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f), save_dir

    return {"status": "error", "equation": "(result.json not found)"}, save_dir
