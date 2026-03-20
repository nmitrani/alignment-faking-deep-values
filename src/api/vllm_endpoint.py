"""Utility for discovering and health-checking a local vLLM classifier server."""

import os
import subprocess
import time
import urllib.request
import urllib.error

_ENDPOINT_FILENAME = ".vllm_endpoint"


def _endpoint_file_path() -> str:
    """Return the path to the endpoint discovery file."""
    # Walk up from this file to the repo root (src/api/ -> repo root)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, _ENDPOINT_FILENAME)


def _is_slurm_job_running(job_id: str) -> bool:
    """Check if a SLURM job is still running via squeue."""
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True, text=True, timeout=10,
        )
        state = result.stdout.strip()
        return state in ("RUNNING", "PENDING", "CONFIGURING")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # squeue not available (not on a SLURM cluster) — assume job is valid
        return True


def discover_endpoint() -> tuple[str, str] | None:
    """Read .vllm_endpoint and return (base_url, job_id), or None if stale/missing."""
    path = _endpoint_file_path()
    if not os.path.exists(path):
        return None

    try:
        with open(path) as f:
            lines = f.read().strip().splitlines()
        if len(lines) < 2:
            return None
        base_url, job_id = lines[0].strip(), lines[1].strip()
    except OSError:
        return None

    # Validate SLURM job is still running
    if job_id != "unknown" and not _is_slurm_job_running(job_id):
        # Stale endpoint file — clean up
        try:
            os.remove(path)
        except OSError:
            pass
        return None

    return base_url, job_id


def check_health(base_url: str, timeout: float = 5.0) -> bool:
    """GET /health on the vLLM server. Returns True if healthy."""
    # base_url is like http://host:port/v1 — health endpoint is at /health
    health_url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def ensure_classifier_endpoint(auto_launch: bool = False) -> str | None:
    """Discover a healthy local vLLM endpoint.

    Returns the base_url if found and healthy, None otherwise.
    If auto_launch=True and no server is found, submits an sbatch job and
    polls until ready (up to 10 minutes).
    """
    # Check env var override first
    env_url = os.environ.get("CLASSIFIER_VLLM_BASE_URL")
    if env_url:
        if check_health(env_url):
            return env_url
        # Env var set but server not healthy — don't fall through to discovery
        print(f"[vLLM] Warning: CLASSIFIER_VLLM_BASE_URL={env_url} is set but server is not healthy")
        return None

    # Try discovery
    result = discover_endpoint()
    if result:
        base_url, _ = result
        if check_health(base_url):
            return base_url

    if not auto_launch:
        return None

    # Auto-launch via sbatch
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sbatch_script = os.path.join(repo_root, "run_on_compute.sbatch")
    serve_script = os.path.join(repo_root, "serve_classifier.sh")

    if not os.path.exists(sbatch_script) or not os.path.exists(serve_script):
        print("[vLLM] Cannot auto-launch: missing run_on_compute.sbatch or serve_classifier.sh")
        return None

    print("[vLLM] No classifier server found. Submitting sbatch job...")
    try:
        subprocess.run(
            ["sbatch", "--gpus=2", "--mem=430G", "--time=24:00:00",
             sbatch_script, f"./{os.path.basename(serve_script)}"],
            cwd=repo_root, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[vLLM] Failed to submit sbatch job: {e}")
        return None

    # Poll for endpoint to appear (up to 10 minutes)
    max_wait = 600
    waited = 0
    while waited < max_wait:
        time.sleep(10)
        waited += 10
        result = discover_endpoint()
        if result:
            base_url, _ = result
            if check_health(base_url):
                print(f"[vLLM] Server ready after {waited}s: {base_url}")
                return base_url

    print(f"[vLLM] Server did not become ready within {max_wait}s")
    return None
