"""
LoRA training job handler for the lora-train-a100 Vast lane.

Full lifecycle:
  1. Download training images from S3 → /data/images/{run_id}/
  2. Write caption .txt sidecar files alongside each image
  3. Write kohya-ss config.toml → /tmp/lora_train_{run_id}/config.toml
  4. Handle resume-from-checkpoint: download latest .safetensors from S3 if requested
  5. Launch:  accelerate launch sd_scripts/train_network.py --config /tmp/lora_train_{run_id}/config.toml
  6. Stream stdout → parse step count → POST progress to backend ≤60 s or every 500 steps
  7. Watch output_dir for new .safetensors checkpoints → upload to S3
  8. Exit 0: upload final LoRA, POST completion callback
  9. Exit non-zero: POST failure callback with stderr tail
  10. CUDA OOM: reduce network_dim to 32, retry once; second OOM → fail
  11. Spot interruption (SIGTERM): re-queue job with resume_from_checkpoint pointing to latest S3 checkpoint

Env vars consumed (set on the Vast instance template):
  S3_ENDPOINT_URL       — endpoint for all S3 operations
  S3_ACCESS_KEY_ID
  S3_SECRET_ACCESS_KEY
  S3_REGION             — default "us-east-1"
  BACKEND_URL           — NudeLab backend base URL for progress/completion callbacks
  SD_SCRIPTS_DIR        — path to sd_scripts repo (default /workspace/kohya_ss/sd_scripts)
  LORA_OUTPUT_BASE      — base dir for training output (default /data/lora_output)
  LORA_IMAGES_BASE      — base dir for training images (default /data/images)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import boto3
import requests
import tomli_w  # kohya-ss config is TOML

try:
    from workers.comfyui_json.s3_boto_resilience import (
        build_s3_boto_config,
        download_file_with_retry,
        upload_file_with_retry,
    )
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "comfyui-json"))
    from s3_boto_resilience import (  # type: ignore[no-redef]
        build_s3_boto_config,
        download_file_with_retry,
        upload_file_with_retry,
    )

_log = logging.getLogger("lora-train")

SD_SCRIPTS_DIR = Path(os.getenv("SD_SCRIPTS_DIR", "/workspace/kohya_ss/sd_scripts"))
LORA_OUTPUT_BASE = Path(os.getenv("LORA_OUTPUT_BASE", "/data/lora_output"))
LORA_IMAGES_BASE = Path(os.getenv("LORA_IMAGES_BASE", "/data/images"))
BACKEND_URL = os.getenv("BACKEND_URL", "")

# Kohya-ss stdout step patterns (handles various output formats)
_STEP_RE = re.compile(
    r"(?:steps?|step)\s*[:\s]\s*(\d+)\s*/\s*(\d+)"
    r"|(\d+)\s*steps?\s*/\s*(\d+)",
    re.IGNORECASE,
)
# OOM markers in stderr
_OOM_RE = re.compile(r"CUDA\s+out\s+of\s+memory|OutOfMemoryError", re.IGNORECASE)

_PROGRESS_INTERVAL_STEPS = 500
_PROGRESS_INTERVAL_SECS = 60.0


def _s3_client(endpoint_url: str | None = None) -> Any:
    endpoint = endpoint_url or os.getenv("S3_ENDPOINT_URL", "")
    return boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
        region_name=os.getenv("S3_REGION", "us-east-1"),
        config=build_s3_boto_config(signature_version="s3v4"),
    )


def _post_progress(
    training_run_id: str,
    progress_percent: int,
    current_step: int,
    total_steps: int,
    status: str = "RUNNING",
) -> None:
    if not BACKEND_URL:
        _log.warning("BACKEND_URL not set; skipping progress callback")
        return
    url = f"{BACKEND_URL.rstrip('/')}/internal/lora-training-progress/{training_run_id}"
    body = {
        "progress_percent": progress_percent,
        "current_step": current_step,
        "total_steps": total_steps,
        "status": status,
    }
    try:
        resp = requests.post(url, json=body, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        _log.warning("Progress callback failed (run=%s step=%d): %s", training_run_id, current_step, exc)


def _post_completion(training_run_id: str, output_lora_url: str) -> None:
    if not BACKEND_URL:
        _log.warning("BACKEND_URL not set; skipping completion callback")
        return
    url = f"{BACKEND_URL.rstrip('/')}/internal/lora-training-callback/{training_run_id}"
    body = {
        "training_run_id": training_run_id,
        "status": "completed",
        "output_lora_url": output_lora_url,
    }
    try:
        resp = requests.post(url, json=body, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        _log.error("Completion callback failed (run=%s): %s", training_run_id, exc)


def _post_failure(training_run_id: str, error: str) -> None:
    if not BACKEND_URL:
        _log.warning("BACKEND_URL not set; skipping failure callback")
        return
    url = f"{BACKEND_URL.rstrip('/')}/internal/lora-training-callback/{training_run_id}"
    body = {
        "training_run_id": training_run_id,
        "status": "failed",
        "error": error,
    }
    try:
        resp = requests.post(url, json=body, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        _log.error("Failure callback failed (run=%s): %s", training_run_id, exc)


def _requeue_for_checkpoint(training_run_id: str, checkpoint_s3_url: str) -> None:
    """Called on SIGTERM/spot interruption. Asks the backend to re-queue with resume_from_checkpoint."""
    if not BACKEND_URL:
        _log.warning("BACKEND_URL not set; cannot re-queue for checkpoint")
        return
    url = f"{BACKEND_URL.rstrip('/')}/internal/lora-training-callback/{training_run_id}"
    body = {
        "training_run_id": training_run_id,
        "status": "interrupted",
        "resume_from_checkpoint": True,
        "resume_checkpoint_path": checkpoint_s3_url,
    }
    try:
        resp = requests.post(url, json=body, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        _log.error("Re-queue callback failed (run=%s): %s", training_run_id, exc)


def _download_training_images(
    dataset_items: list[dict],
    image_dir: Path,
    s3_endpoint_url: str,
    s3_bucket: str,
) -> None:
    """Download training images to image_dir; write .txt caption sidecars."""
    image_dir.mkdir(parents=True, exist_ok=True)
    client = _s3_client(s3_endpoint_url)
    for i, item in enumerate(dataset_items):
        bucket = item.get("s3_bucket") or s3_bucket
        key = item.get("s3_key") or item.get("key") or ""
        caption = item.get("caption") or ""
        if not key:
            _log.warning("dataset_items[%d] missing s3_key; skipping", i)
            continue
        filename = Path(key).name or f"image_{i:05d}.jpg"
        local_img = image_dir / filename
        _log.info("Downloading %s/%s -> %s", bucket, key, local_img)
        download_file_with_retry(client, bucket, key, str(local_img))
        stem = local_img.stem
        (image_dir / f"{stem}.txt").write_text(caption, encoding="utf-8")
    _log.info("Downloaded %d training images to %s", len(dataset_items), image_dir)


def _write_config_toml(
    config_path: Path,
    training_config: dict,
    image_dir: Path,
    output_dir: Path,
    resume_checkpoint_local: str | None = None,
) -> None:
    """Write kohya-ss train_network.py config.toml from training_config payload fields.

    Payload key mapping (confirmed NUD-44 / NUD-63 contract):
      training_config["steps"]         → max_train_steps
      training_config["framework"]     → stripped (internal, not a kohya-ss field)
    All other keys are passed through verbatim.
    """
    cfg = dict(training_config)

    # Normalize payload key → kohya-ss key
    if "steps" in cfg and "max_train_steps" not in cfg:
        cfg["max_train_steps"] = cfg.pop("steps")
    else:
        cfg.pop("steps", None)  # remove if max_train_steps already present

    # Strip internal-only keys that kohya-ss does not understand
    cfg.pop("framework", None)

    cfg.setdefault("train_data_dir", str(image_dir))
    cfg.setdefault("output_dir", str(output_dir))
    cfg.setdefault("output_name", "lora_output")
    cfg.setdefault("save_model_as", "safetensors")
    cfg.setdefault("network_module", "networks.lora")
    cfg.setdefault("mixed_precision", "bf16")
    cfg.setdefault("save_precision", "bf16")
    cfg.setdefault("logging_dir", str(output_dir / "logs"))
    if resume_checkpoint_local:
        cfg["resume"] = resume_checkpoint_local

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "wb") as f:
        tomli_w.dump(cfg, f)
    _log.info("Wrote config.toml -> %s", config_path)


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the most recently modified .safetensors checkpoint in output_dir, or None."""
    checkpoints = sorted(
        output_dir.glob("*.safetensors"),
        key=lambda p: p.stat().st_mtime,
    )
    return checkpoints[-1] if checkpoints else None


def _upload_checkpoint(
    client: Any,
    local_path: Path,
    checkpoint_bucket: str,
    checkpoint_prefix: str,
    step: int | None = None,
) -> str:
    """Upload checkpoint to S3; return the s3:// URL."""
    filename = local_path.name
    if step is not None:
        filename = f"step-{step}.safetensors"
    key = f"{checkpoint_prefix.strip('/')}/{filename}"
    _log.info("Uploading checkpoint %s -> s3://%s/%s", local_path, checkpoint_bucket, key)
    upload_file_with_retry(client, str(local_path), checkpoint_bucket, key)
    return f"s3://{checkpoint_bucket}/{key}"


def _parse_step(line: str) -> tuple[int, int] | None:
    """Extract (current_step, total_steps) from a kohya-ss log line, or None."""
    m = _STEP_RE.search(line)
    if m:
        if m.group(1) and m.group(2):
            return int(m.group(1)), int(m.group(2))
        if m.group(3) and m.group(4):
            return int(m.group(3)), int(m.group(4))
    return None


def _run_training(
    config_path: Path,
    output_dir: Path,
    training_run_id: str,
    total_steps: int,
    checkpoint_bucket: str,
    checkpoint_prefix: str,
    s3_endpoint_url: str,
) -> subprocess.CompletedProcess[bytes]:
    """
    Launch training subprocess; stream stdout; post progress + upload checkpoints.
    Returns the CompletedProcess on completion (or raises on SIGTERM/OOM).

    OOM detection: raises RuntimeError("OOM") on CUDA out of memory.
    """
    train_script = SD_SCRIPTS_DIR / "train_network.py"
    cmd = [
        sys.executable,
        "-u",  # unbuffered stdout
        str(train_script),
        "--config",
        str(config_path),
    ]
    _log.info("Launching training: %s", " ".join(cmd))

    s3_client = _s3_client(s3_endpoint_url)
    uploaded_checkpoints: set[str] = set()
    stdout_lines: list[str] = []
    stderr_buf: list[str] = []

    last_progress_step = 0
    last_progress_time = time.monotonic()
    current_step = 0

    interrupted = threading.Event()
    original_sigterm = signal.getsignal(signal.SIGTERM)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def _on_sigterm(signum: int, frame: Any) -> None:
        _log.warning("SIGTERM received — spot interruption; flagging interrupted")
        interrupted.set()
        proc.terminate()

    signal.signal(signal.SIGTERM, _on_sigterm)

    def _monitor_checkpoints() -> None:
        """Background thread: watch for new .safetensors in output_dir and upload."""
        while proc.poll() is None or not interrupted.is_set():
            for ckpt in sorted(output_dir.glob("*.safetensors")):
                key = str(ckpt)
                if key in uploaded_checkpoints:
                    continue
                # Extract step number from filename (e.g. lora_output-000500.safetensors)
                step_match = re.search(r"-(\d+)\.safetensors$", ckpt.name)
                step_num = int(step_match.group(1)) if step_match else None
                try:
                    _upload_checkpoint(
                        s3_client, ckpt, checkpoint_bucket, checkpoint_prefix, step=step_num
                    )
                    uploaded_checkpoints.add(key)
                except Exception as exc:
                    _log.warning("Checkpoint upload failed (%s): %s", ckpt.name, exc)
            time.sleep(10)

    ckpt_thread = threading.Thread(target=_monitor_checkpoints, daemon=True)
    ckpt_thread.start()

    oom_detected = False
    assert proc.stdout is not None
    assert proc.stderr is not None

    # Read stderr in a background thread to avoid pipe deadlock
    def _read_stderr() -> None:
        for line in proc.stderr:
            line = line.rstrip()
            stderr_buf.append(line)
            if _OOM_RE.search(line):
                nonlocal oom_detected
                oom_detected = True
            _log.debug("[stderr] %s", line)

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    for line in proc.stdout:
        line = line.rstrip()
        stdout_lines.append(line)
        _log.info("[train] %s", line)

        parsed = _parse_step(line)
        if parsed:
            current_step, _ = parsed

        now = time.monotonic()
        steps_since = current_step - last_progress_step
        secs_since = now - last_progress_time
        if steps_since >= _PROGRESS_INTERVAL_STEPS or secs_since >= _PROGRESS_INTERVAL_SECS:
            pct = int(current_step / total_steps * 100) if total_steps else 0
            _post_progress(training_run_id, pct, current_step, total_steps)
            last_progress_step = current_step
            last_progress_time = now

    proc.wait()
    stderr_thread.join(timeout=5)
    signal.signal(signal.SIGTERM, original_sigterm)

    if interrupted.is_set():
        # Find latest uploaded checkpoint URL for re-queue
        latest_ckpt = _find_latest_checkpoint(output_dir)
        ckpt_url = ""
        if latest_ckpt and str(latest_ckpt) in uploaded_checkpoints:
            step_match = re.search(r"-(\d+)\.safetensors$", latest_ckpt.name)
            step_num = int(step_match.group(1)) if step_match else None
            ckpt_key = f"{checkpoint_prefix.strip('/')}/step-{step_num}.safetensors" if step_num else f"{checkpoint_prefix.strip('/')}/{latest_ckpt.name}"
            ckpt_url = f"s3://{checkpoint_bucket}/{ckpt_key}"
        _requeue_for_checkpoint(training_run_id, ckpt_url)
        raise RuntimeError("INTERRUPTED")

    if oom_detected:
        raise RuntimeError("OOM")

    stderr_tail = "\n".join(stderr_buf[-50:])
    return subprocess.CompletedProcess(
        cmd, proc.returncode, stdout="\n".join(stdout_lines), stderr=stderr_tail
    )


def run_lora_training(payload: dict) -> dict:
    """
    Entry point: called by the Vast SDK request_parser with the lora_train job payload.

    Payload fields (from NUD-44 item 5 / issue NUD-66):
      training_run_id         str
      dataset_items           list[{s3_key, s3_bucket?, caption}]
      training_config         dict  — kohya-ss train_network.py config fields
      s3_bucket               str   — default bucket for training images
      s3_endpoint_url         str
      checkpoint_bucket       str
      checkpoint_prefix       str
      output_destination      {bucket, key}  — final LoRA artifact
      resume_from_checkpoint  bool
      resume_checkpoint_path  str   — S3 path to resume from (when resume_from_checkpoint=True)
      backend_url             str   — override BACKEND_URL env var
    """
    global BACKEND_URL

    # Allow per-job backend URL override
    job_backend_url = (payload.get("backend_url") or "").strip()
    if job_backend_url:
        BACKEND_URL = job_backend_url

    training_run_id: str = payload.get("training_run_id") or ""
    if not training_run_id:
        raise ValueError("lora_train job missing training_run_id")

    dataset_items: list[dict] = payload.get("dataset_items") or []
    training_config: dict = dict(payload.get("training_config") or {})
    s3_bucket: str = payload.get("s3_bucket") or ""
    s3_endpoint_url: str = payload.get("s3_endpoint_url") or os.getenv("S3_ENDPOINT_URL", "")
    checkpoint_bucket: str = payload.get("checkpoint_bucket") or s3_bucket
    checkpoint_prefix: str = payload.get("checkpoint_prefix") or f"lora-checkpoints/{training_run_id}"
    output_destination: dict = payload.get("output_destination") or {}
    resume_from_checkpoint: bool = bool(payload.get("resume_from_checkpoint"))
    resume_checkpoint_path: str = payload.get("resume_checkpoint_path") or ""
    # NUD-44/NUD-63 confirmed key is "steps"; accept "max_train_steps" as fallback
    total_steps: int = int(
        training_config.get("steps") or training_config.get("max_train_steps") or 1000
    )

    image_dir = LORA_IMAGES_BASE / training_run_id
    output_dir = LORA_OUTPUT_BASE / training_run_id
    config_dir = Path(f"/tmp/lora_train_{training_run_id}")
    config_path = config_dir / "config.toml"

    _log.info(
        "Starting lora_train run=%s steps=%d resume=%s",
        training_run_id, total_steps, resume_from_checkpoint,
    )

    try:
        # 1. Download training images + write caption sidecars
        _download_training_images(dataset_items, image_dir, s3_endpoint_url, s3_bucket)

        # 2. Handle resume-from-checkpoint
        resume_checkpoint_local: str | None = None
        if resume_from_checkpoint and resume_checkpoint_path:
            _log.info("Downloading resume checkpoint: %s", resume_checkpoint_path)
            # Parse s3://bucket/key
            ckpt_path_stripped = resume_checkpoint_path.lstrip("s3://")
            parts = ckpt_path_stripped.split("/", 1)
            ckpt_bucket, ckpt_key = (parts[0], parts[1]) if len(parts) == 2 else (checkpoint_bucket, ckpt_path_stripped)
            local_ckpt = config_dir / "resume.safetensors"
            local_ckpt.parent.mkdir(parents=True, exist_ok=True)
            client = _s3_client(s3_endpoint_url)
            download_file_with_retry(client, ckpt_bucket, ckpt_key, str(local_ckpt))
            resume_checkpoint_local = str(local_ckpt)
            _log.info("Resume checkpoint downloaded to %s", local_ckpt)

        # 3. Write config.toml
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_config_toml(
            config_path, training_config, image_dir, output_dir, resume_checkpoint_local
        )

        # 4 + 5 + 6. Launch training (with CUDA OOM retry)
        oom_retried = False
        while True:
            try:
                result = _run_training(
                    config_path,
                    output_dir,
                    training_run_id,
                    total_steps,
                    checkpoint_bucket,
                    checkpoint_prefix,
                    s3_endpoint_url,
                )
                break
            except RuntimeError as e:
                if str(e) == "OOM" and not oom_retried:
                    _log.warning("CUDA OOM detected; reducing network_dim to 32 and retrying")
                    training_config["network_dim"] = 32
                    _write_config_toml(
                        config_path, training_config, image_dir, output_dir, resume_checkpoint_local
                    )
                    oom_retried = True
                    continue
                raise

        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip()
            error_msg = f"Training exited {result.returncode}: {stderr_tail[-500:]}"
            _log.error("%s", error_msg)
            _post_failure(training_run_id, error_msg)
            return {"status": "failed", "error": error_msg}

        # 7. Upload final LoRA artifact
        final_ckpt = _find_latest_checkpoint(output_dir)
        if not final_ckpt:
            err = "Training completed but no .safetensors found in output_dir"
            _log.error("%s", err)
            _post_failure(training_run_id, err)
            return {"status": "failed", "error": err}

        out_bucket = output_destination.get("bucket") or checkpoint_bucket
        out_key = output_destination.get("key") or f"lora/{training_run_id}/final.safetensors"
        client = _s3_client(s3_endpoint_url)
        _log.info("Uploading final LoRA %s -> s3://%s/%s", final_ckpt, out_bucket, out_key)
        upload_file_with_retry(client, str(final_ckpt), out_bucket, out_key)
        output_lora_url = f"s3://{out_bucket}/{out_key}"

        # 8. Post completion callback
        _post_completion(training_run_id, output_lora_url)
        _log.info("lora_train run=%s COMPLETED output=%s", training_run_id, output_lora_url)
        return {
            "status": "completed",
            "training_run_id": training_run_id,
            "output_lora_url": output_lora_url,
        }

    except RuntimeError as exc:
        if str(exc) == "INTERRUPTED":
            _log.warning("lora_train run=%s spot-interrupted; re-queue sent", training_run_id)
            return {"status": "interrupted", "training_run_id": training_run_id}
        if str(exc) == "OOM":
            err = "CUDA OOM after retry with network_dim=32"
            _log.error("lora_train run=%s %s", training_run_id, err)
            _post_failure(training_run_id, err)
            return {"status": "failed", "error": err}
        err = str(exc)
        _log.error("lora_train run=%s unexpected error: %s", training_run_id, err)
        _post_failure(training_run_id, err)
        return {"status": "failed", "error": err}
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        _log.error("lora_train run=%s fatal: %s", training_run_id, err, exc_info=True)
        _post_failure(training_run_id, err)
        return {"status": "failed", "error": err}
    finally:
        # Clean up per-job scratch dirs
        for d in (image_dir, config_dir):
            try:
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
