"""
Vast PyWorker for LoRA training jobs — lora-train-a100 lane.

Unlike the comfyui-json worker, this worker has NO persistent model server.
The Vast SDK Worker is configured with model_server_url=None so it does not
wait for an external backend. All training logic runs directly inside the
request_parser (handler.run_lora_training).

Lane: lora-train-a100
Backend env var: BACKEND=lora-train
Vast instance: A100, comfyui: false

Env vars required on the Vast template:
  BACKEND            = lora-train
  SD_SCRIPTS_DIR     = /workspace/kohya_ss/sd_scripts
  BACKEND_URL        = https://<nudelab-backend>/
  S3_ENDPOINT_URL    = ...
  S3_ACCESS_KEY_ID   = ...
  S3_SECRET_ACCESS_KEY = ...
  S3_REGION          = us-east-1   (or appropriate region)
  LORA_OUTPUT_BASE   = /data/lora_output   (optional, has default)
  LORA_IMAGES_BASE   = /data/images        (optional, has default)
"""

import logging
import os

from vastai import HandlerConfig, Worker, WorkerConfig

from .handler import run_lora_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
_log = logging.getLogger("lora-train.worker")

# Training jobs are long-running (60–90 min); allow adequate queue wait.
# max_queue_time: how long a job waits in the Vast queue before rejection.
_MAX_QUEUE_TIME = float(os.getenv("LORA_MAX_QUEUE_TIME", "300"))


def workload_calculator(payload: dict) -> float:
    """
    Declared Vast workload cost for billing/routing.

    A training job declares total_steps as a proxy for cost.
    Falls back to env var VAST_WORKLOAD_UNITS_LORA_TRAIN or a safe default.
    """
    inp = payload.get("input") or payload
    training_config = inp.get("training_config") or {}
    max_steps = training_config.get("max_train_steps")
    if max_steps:
        try:
            steps = float(max_steps)
            return max(1.0, steps)
        except (TypeError, ValueError):
            pass

    env_val = os.getenv("VAST_WORKLOAD_UNITS_LORA_TRAIN")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass

    return 5000.0  # default: ~5000 steps worth of cost


def request_parser(payload: dict) -> dict:
    """
    Route incoming job to the lora training handler.

    The Vast SDK calls this function with the job payload.
    For lora-train, there is no separate model server — the handler
    performs all training work and returns the result dict.

    Because model_server_url=None, the SDK must not try to forward
    the return value to a backend service; the return value IS the response.
    """
    inp = payload.get("input") or payload
    job_type = (inp.get("job_type") or "lora_train").strip()
    if job_type != "lora_train":
        raise ValueError(
            f"lora-train worker received unexpected job_type={job_type!r}; "
            "only lora_train is supported on this lane"
        )
    return run_lora_training(inp)


worker_config = WorkerConfig(
    # No persistent model backend — training runs inline in request_parser.
    # model_server_url=None disables model-server health monitoring.
    model_server_url=None,
    model_log_file=None,
    handlers=[
        HandlerConfig(
            route="/lora-train",
            allow_parallel_requests=False,
            max_queue_time=_MAX_QUEUE_TIME,
            workload_calculator=workload_calculator,
            request_parser=request_parser,
        ),
    ],
)

if __name__ == "__main__":
    _log.info("Starting lora-train Vast worker (lane=lora-train-a100, no ComfyUI)")
    Worker(worker_config).run()
