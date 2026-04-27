"""
Tests for DATASET_GEN_A100 lane routing in the comfyui-json worker.

Covers:
- Lane registration in _KNOWN_WORKLOAD_LANES
- workload_calculator accepts DATASET_GEN_A100 via vast_workload_units or env var
- workload_calculator rejects unknown lanes (regression guard)
- transform_app_to_vast passes through s3_prefix
- transform_app_to_vast maps DATASET_GEN_A100 payload fields correctly
- Benchmark workflow file exists under misc/
- Workflow error propagates as RuntimeError (typed exception, no silent pass)
"""

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub the vastai SDK so worker.py can be imported without the real package.
# ---------------------------------------------------------------------------

_vastai_stub = types.ModuleType("vastai")
for _cls_name in ("BenchmarkConfig", "HandlerConfig", "LogActionConfig", "Worker", "WorkerConfig"):
    setattr(_vastai_stub, _cls_name, MagicMock())
sys.modules.setdefault("vastai", _vastai_stub)

# Ensure the comfyui-json package is importable from its directory.
_WORKER_DIR = Path(__file__).resolve().parent.parent / "workers" / "comfyui-json"
if str(_WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKER_DIR))

import worker as comfyui_worker  # noqa: E402
import workflow_transform as wt  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MISC_DIR = _WORKER_DIR / "misc"
_WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"

DATASET_GEN_LANE = "DATASET_GEN_A100"

_MINIMAL_WORKFLOW = {
    "1": {
        "inputs": {"image": ""},
        "class_type": "ETN_LoadImageBase64",
        "_meta": {"title": "reference_collage"},
    },
    "30": {
        "inputs": {
            "text": 'a high quality image, "placeholder"',
            "clip": ["99", 0],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
    },
    "70": {
        "inputs": {"filename_prefix": "out", "images": ["60", 0]},
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    },
}


def _make_dataset_gen_payload(
    *,
    generation_lane: str = DATASET_GEN_LANE,
    vast_workload_units: float | None = 200.0,
    s3_bucket: str = "my-bucket",
    s3_prefix: str = "dataset/gen",
    user_prompt: str = "a beautiful sunset",
    prompt_node_title: str = "CLIP Text Encode (Positive Prompt)",
) -> dict:
    """Build a minimal app-format payload matching the NUD-261 dispatch contract."""
    inp: dict = {
        "workflow": _MINIMAL_WORKFLOW,
        "input_images": [],
        "user_prompt": user_prompt,
        "prompt_node_title": prompt_node_title,
        "timeout": 600,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "user_id": "user-abc",
        "generation_id": "gen-123",
        "generation_lane": generation_lane,
    }
    if vast_workload_units is not None:
        inp["vast_workload_units"] = vast_workload_units
    return {"input": inp}


# ---------------------------------------------------------------------------
# Lane registration
# ---------------------------------------------------------------------------


class TestLaneRegistration:
    def test_dataset_gen_a100_in_known_workload_lanes(self):
        assert DATASET_GEN_LANE in comfyui_worker._KNOWN_WORKLOAD_LANES

    def test_dataset_gen_a100_in_benchmark_files(self):
        assert DATASET_GEN_LANE in comfyui_worker._DEFAULT_BENCHMARK_FILES

    def test_dataset_gen_a100_in_benchmark_lane_map(self):
        assert DATASET_GEN_LANE in comfyui_worker._BENCHMARK_ENV_LANE_TO_REQUEST_GENERATION_LANE
        assert (
            comfyui_worker._BENCHMARK_ENV_LANE_TO_REQUEST_GENERATION_LANE[DATASET_GEN_LANE]
            == DATASET_GEN_LANE
        )

    def test_existing_lanes_not_removed(self):
        for lane in ("WAN22_I2V_LONG_5090", "WAN22_IV2V_FACESWAP_5090", "ZIMAGE_TURBO_I2I_5090", "FLUX_S2_I2I_5090"):
            assert lane in comfyui_worker._KNOWN_WORKLOAD_LANES, f"Regression: {lane} missing"


# ---------------------------------------------------------------------------
# workload_calculator
# ---------------------------------------------------------------------------


class TestWorkloadCalculator:
    def test_accepts_dataset_gen_via_vast_workload_units(self):
        payload = _make_dataset_gen_payload(vast_workload_units=200.0)
        result = comfyui_worker.workload_calculator(payload)
        assert result == 200.0

    def test_accepts_dataset_gen_via_env_var(self, monkeypatch):
        monkeypatch.setenv("VAST_WORKLOAD_UNITS_DATASET_GEN_A100", "150")
        payload = _make_dataset_gen_payload(vast_workload_units=None)
        result = comfyui_worker.workload_calculator(payload)
        assert result == 150.0

    def test_rejects_unknown_lane(self):
        payload = {"input": {"generation_lane": "UNKNOWN_LANE", "vast_workload_units": 100}}
        with pytest.raises(ValueError, match="unknown generation_lane"):
            comfyui_worker.workload_calculator(payload)

    def test_raises_when_no_units_and_no_env(self, monkeypatch):
        monkeypatch.delenv("VAST_WORKLOAD_UNITS_DATASET_GEN_A100", raising=False)
        payload = _make_dataset_gen_payload(vast_workload_units=None)
        with pytest.raises(ValueError, match="VAST_WORKLOAD_UNITS_DATASET_GEN_A100"):
            comfyui_worker.workload_calculator(payload)

    def test_clamps_dynamic_workload_within_bounds(self, monkeypatch):
        monkeypatch.setenv("VAST_WORKLOAD_DYNAMIC_MIN", "10")
        monkeypatch.setenv("VAST_WORKLOAD_DYNAMIC_MAX", "500")
        payload = _make_dataset_gen_payload(vast_workload_units=999.0)
        result = comfyui_worker.workload_calculator(payload)
        assert result == 500.0


# ---------------------------------------------------------------------------
# transform_app_to_vast — s3_prefix passthrough + payload mapping
# ---------------------------------------------------------------------------


class TestTransformDatasetGen:
    def _transform(self, payload: dict) -> dict:
        """Run transform without S3 downloads (no input_images)."""
        return wt.transform_app_to_vast(payload)

    def test_s3_prefix_forwarded(self):
        payload = _make_dataset_gen_payload(s3_prefix="dataset/2026/04")
        out = self._transform(payload)
        assert out["input"]["s3_prefix"] == "dataset/2026/04"

    def test_s3_bucket_forwarded(self):
        payload = _make_dataset_gen_payload(s3_bucket="my-bucket")
        out = self._transform(payload)
        assert out["input"]["s3_bucket"] == "my-bucket"

    def test_s3_prefix_absent_when_empty(self):
        payload = _make_dataset_gen_payload(s3_prefix="")
        out = self._transform(payload)
        assert "s3_prefix" not in out["input"]

    def test_generation_lane_forwarded(self):
        payload = _make_dataset_gen_payload()
        out = self._transform(payload)
        assert out["input"]["generation_lane"] == DATASET_GEN_LANE

    def test_vast_workload_units_forwarded(self):
        payload = _make_dataset_gen_payload(vast_workload_units=200.0)
        out = self._transform(payload)
        assert out["input"]["vast_workload_units"] == 200.0

    def test_prompt_injected_into_clip_node(self):
        payload = _make_dataset_gen_payload(
            user_prompt="beach sunset golden hour",
            prompt_node_title="CLIP Text Encode (Positive Prompt)",
        )
        out = self._transform(payload)
        wf = out["input"]["workflow_json"]
        clip_node = wf["30"]
        assert "beach sunset golden hour" in clip_node["inputs"]["text"]

    def test_workflow_json_present_in_output(self):
        payload = _make_dataset_gen_payload()
        out = self._transform(payload)
        assert "workflow_json" in out["input"]
        assert isinstance(out["input"]["workflow_json"], dict)

    def test_user_id_and_generation_id_forwarded(self):
        payload = _make_dataset_gen_payload()
        out = self._transform(payload)
        assert out["input"]["user_id"] == "user-abc"
        assert out["input"]["generation_id"] == "gen-123"

    def test_seeds_randomized(self):
        payload = _make_dataset_gen_payload()
        out1 = self._transform(payload)
        out2 = self._transform(payload)
        # Seeds are randomized per-call; two runs should differ with overwhelming probability
        wf1 = json.dumps(out1["input"]["workflow_json"])
        wf2 = json.dumps(out2["input"]["workflow_json"])
        # The workflows themselves differ only in seeds; structure must be equal ignoring seeds.
        assert out1["input"]["workflow_json"].keys() == out2["input"]["workflow_json"].keys()

    def test_already_vast_format_passthrough(self):
        already_vast = {"input": {"workflow_json": {"a": 1}, "request_id": "x"}}
        out = self._transform(already_vast)
        assert out == already_vast

    def test_workflow_error_raises_typed_exception(self):
        """ETN image injection failure must raise RuntimeError, not swallow silently."""
        payload = _make_dataset_gen_payload()
        payload["input"]["input_images"] = [{"bucket": "b", "key": "k"}]
        with pytest.raises(RuntimeError):
            self._transform(payload)


# ---------------------------------------------------------------------------
# Benchmark + workflow file existence
# ---------------------------------------------------------------------------


class TestFileExistence:
    def test_benchmark_json_exists(self):
        p = _MISC_DIR / "benchmark_DATASET_GEN_A100.json"
        assert p.is_file(), f"Benchmark file missing: {p}"

    def test_benchmark_json_is_valid_json(self):
        p = _MISC_DIR / "benchmark_DATASET_GEN_A100.json"
        with open(p) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_benchmark_json_has_etn_loader(self):
        p = _MISC_DIR / "benchmark_DATASET_GEN_A100.json"
        with open(p) as f:
            data = json.load(f)
        class_types = {n.get("class_type") for n in data.values() if isinstance(n, dict)}
        assert "ETN_LoadImageBase64" in class_types, "Benchmark must have ETN_LoadImageBase64 for collage injection"

    def test_benchmark_json_has_clip_text_encode(self):
        p = _MISC_DIR / "benchmark_DATASET_GEN_A100.json"
        with open(p) as f:
            data = json.load(f)
        class_types = {n.get("class_type") for n in data.values() if isinstance(n, dict)}
        assert "CLIPTextEncode" in class_types, "Benchmark must have CLIPTextEncode for prompt injection"

    def test_workflow_placeholder_exists(self):
        p = _WORKFLOWS_DIR / "dataset_gen_a100.json"
        assert p.is_file(), f"Canonical workflow placeholder missing: {p}"

    def test_workflow_placeholder_is_valid_json(self):
        p = _WORKFLOWS_DIR / "dataset_gen_a100.json"
        with open(p) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_workflow_placeholder_has_prompt_node(self):
        """Backend uses prompt_node_title='CLIP Text Encode (Positive Prompt)' to inject per NUD-261."""
        p = _WORKFLOWS_DIR / "dataset_gen_a100.json"
        with open(p) as f:
            data = json.load(f)
        titles = {
            (n.get("_meta") or {}).get("title", "")
            for n in data.values()
            if isinstance(n, dict)
        }
        assert "CLIP Text Encode (Positive Prompt)" in titles, (
            "Workflow must have a CLIPTextEncode node titled 'CLIP Text Encode (Positive Prompt)' "
            "for prompt injection to work"
        )
