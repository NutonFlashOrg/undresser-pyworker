"""
Tests for reference_image_url wiring in workflow_transform.py.

Covers:
- reference_image_url present → image downloaded and injected into LoadImage "Reference Image" node
- reference_image_url absent → transform output unchanged (no regression)
- Invalid URL → raises RuntimeError with clear message
"""

import copy
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Stub vastai SDK so worker.py can be imported without the real package.
_vastai_stub = types.ModuleType("vastai")
for _cls in ("BenchmarkConfig", "HandlerConfig", "LogActionConfig", "Worker", "WorkerConfig"):
    setattr(_vastai_stub, _cls, MagicMock())
sys.modules.setdefault("vastai", _vastai_stub)

_WORKER_DIR = Path(__file__).resolve().parent.parent / "workers" / "comfyui-json"
if str(_WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKER_DIR))

import workflow_transform as wt  # noqa: E402


# ---------------------------------------------------------------------------
# IC-LoRA minimal workflow — LoadImage "Reference Image" + standard FLUX nodes
# ---------------------------------------------------------------------------

_IC_LORA_WORKFLOW = {
    "1": {
        "inputs": {"image": "placeholder.png", "upload": "image"},
        "class_type": "LoadImage",
        "_meta": {"title": "Reference Image"},
    },
    "10": {
        "inputs": {"text": "portrait photo", "clip": ["20", 1]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "Positive Prompt"},
    },
    "20": {
        "inputs": {"ckpt_name": "flux1-schnell-fp8.safetensors"},
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Checkpoint"},
    },
    "50": {
        "inputs": {
            "model": ["20", 0],
            "positive": ["10", 0],
            "negative": ["11", 0],
            "latent_image": ["40", 0],
            "seed": 0,
            "steps": 4,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
    },
    "70": {
        "inputs": {"filename_prefix": "out", "images": ["60", 0]},
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    },
}


def _make_payload(*, reference_image_url: str | None = None) -> dict:
    inp: dict = {
        "workflow": copy.deepcopy(_IC_LORA_WORKFLOW),
        "input_images": [],
        "user_id": "u1",
        "generation_id": "g1",
        "timeout": 60,
    }
    if reference_image_url is not None:
        inp["reference_image_url"] = reference_image_url
    return {"input": inp}


def _workflow_from_result(result: dict) -> dict:
    return result["input"]["workflow_json"]


# ---------------------------------------------------------------------------
# Test: reference_image_url present → injected into LoadImage "Reference Image"
# ---------------------------------------------------------------------------

def test_reference_image_url_injected_into_load_image_node(tmp_path):
    """reference_image_url present → downloaded image path set on LoadImage 'Reference Image' node."""
    fake_image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64  # minimal PNG-like bytes

    def fake_urlretrieve(url, local_path):
        Path(local_path).write_bytes(fake_image_bytes)

    payload = _make_payload(reference_image_url="https://cdn.example.com/ref/photo.png")

    with (
        patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve),
        patch.object(wt, "_comfy_input_root", return_value=tmp_path),
    ):
        result = wt.transform_app_to_vast(payload)

    wf = _workflow_from_result(result)

    ref_node = next(
        (n for n in wf.values()
         if isinstance(n, dict)
         and n.get("class_type") == "LoadImage"
         and ((n.get("_meta") or {}).get("title") or "").strip() == "Reference Image"),
        None,
    )
    assert ref_node is not None, "LoadImage 'Reference Image' node must exist in workflow"
    image_path = (ref_node.get("inputs") or {}).get("image")
    assert image_path is not None, "image input must be set on Reference Image node"
    assert "reference_image" in str(image_path), (
        f"Expected path to reference_image file, got: {image_path!r}"
    )


# ---------------------------------------------------------------------------
# Test: reference_image_url absent → no change to workflow (no regression)
# ---------------------------------------------------------------------------

def test_reference_image_url_absent_no_regression():
    """reference_image_url absent → LoadImage 'Reference Image' node untouched."""
    payload = _make_payload()
    result = wt.transform_app_to_vast(payload)
    wf = _workflow_from_result(result)

    ref_node = next(
        (n for n in wf.values()
         if isinstance(n, dict)
         and n.get("class_type") == "LoadImage"
         and ((n.get("_meta") or {}).get("title") or "").strip() == "Reference Image"),
        None,
    )
    assert ref_node is not None
    assert ref_node["inputs"]["image"] == "placeholder.png", (
        "LoadImage image input must remain unchanged when reference_image_url absent"
    )


# ---------------------------------------------------------------------------
# Test: invalid URL → RuntimeError with clear message
# ---------------------------------------------------------------------------

def test_reference_image_url_invalid_raises_runtime_error(tmp_path):
    """Invalid/unreachable reference_image_url → RuntimeError with URL in message."""
    import urllib.error

    def fake_urlretrieve_fail(url, local_path):
        raise urllib.error.URLError("connection refused")

    payload = _make_payload(reference_image_url="https://invalid.example.com/bad.png")

    with (
        patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve_fail),
        patch.object(wt, "_comfy_input_root", return_value=tmp_path),
    ):
        with pytest.raises(RuntimeError, match="reference_image_url"):
            wt.transform_app_to_vast(payload)


# ---------------------------------------------------------------------------
# Unit test: _download_reference_image_url helper directly
# ---------------------------------------------------------------------------

def test_download_reference_image_url_extension_detection(tmp_path):
    """Helper uses extension from URL path; falls back to .png for unknown extensions."""
    fake_bytes = b"fake-image-data"

    def fake_urlretrieve(url, local_path):
        Path(local_path).write_bytes(fake_bytes)

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
        p = wt._download_reference_image_url(
            "https://cdn.example.com/images/face.jpg", tmp_path
        )
    assert p.suffix == ".jpg"
    assert p.read_bytes() == fake_bytes


def test_download_reference_image_url_unknown_ext_defaults_png(tmp_path):
    """Unknown extension defaults to .png."""
    def fake_urlretrieve(url, local_path):
        Path(local_path).write_bytes(b"data")

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
        p = wt._download_reference_image_url(
            "https://cdn.example.com/images/face.avif", tmp_path
        )
    assert p.suffix == ".png"


def test_download_reference_image_url_failure_raises_runtime_error(tmp_path):
    """Download failure wraps original exception in RuntimeError."""
    import urllib.error

    def fake_urlretrieve(url, local_path):
        raise urllib.error.URLError("timeout")

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
        with pytest.raises(RuntimeError, match="Failed to download reference_image_url"):
            wt._download_reference_image_url("https://bad.example.com/x.png", tmp_path)
