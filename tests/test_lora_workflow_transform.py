"""
Tests for lora_checkpoint + lora_trigger_word handling in workflow_transform.

Covers:
- Both fields present → LoraLoader node inserted + trigger word prepended to positive prompt
- lora_checkpoint only (no trigger word) → LoraLoader inserted, prompt text unchanged
- Neither field present → transform output identical to baseline (no regression)
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
# Shared test workflow
# ---------------------------------------------------------------------------

_POSITIVE_TITLE = "CLIP Text Encode (Positive Prompt)"
_NEGATIVE_TITLE = "CLIP Text Encode (Negative Prompt)"

_MINIMAL_WORKFLOW = {
    "20": {
        "inputs": {"ckpt_name": "model.safetensors"},
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Checkpoint"},
    },
    "30": {
        "inputs": {"text": "high quality image", "clip": ["20", 1]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": _POSITIVE_TITLE},
    },
    "31": {
        "inputs": {"text": "blurry, ugly", "clip": ["20", 1]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": _NEGATIVE_TITLE},
    },
    "50": {
        "inputs": {
            "model": ["20", 0],
            "positive": ["30", 0],
            "negative": ["31", 0],
            "latent_image": ["40", 0],
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
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


def _make_payload(
    *,
    lora_checkpoint: dict | None = None,
    lora_trigger_word: str | None = None,
    prompt_node_title: str = _POSITIVE_TITLE,
) -> dict:
    inp: dict = {
        "workflow": copy.deepcopy(_MINIMAL_WORKFLOW),
        "input_images": [],
        "user_id": "u1",
        "generation_id": "g1",
        "timeout": 60,
        "prompt_node_title": prompt_node_title,
    }
    if lora_checkpoint is not None:
        inp["lora_checkpoint"] = lora_checkpoint
    if lora_trigger_word is not None:
        inp["lora_trigger_word"] = lora_trigger_word
    return {"input": inp}


# ---------------------------------------------------------------------------
# Helper: run transform with S3 mocked out
# ---------------------------------------------------------------------------

def _run_transform(payload: dict, lora_local_path: str = "/tmp/lora/char.safetensors") -> dict:
    """Run transform_app_to_vast with S3 download mocked."""

    def fake_download_lora(lora_ref, scratch_dir):
        p = Path(lora_local_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake safetensors")
        return p

    with patch.object(wt, "_download_lora_checkpoint", side_effect=fake_download_lora):
        return wt.transform_app_to_vast(payload)


def _workflow_from_result(result: dict) -> dict:
    return result["input"]["workflow_json"]


# ---------------------------------------------------------------------------
# Test: neither lora_checkpoint nor lora_trigger_word → no regression
# ---------------------------------------------------------------------------

def test_no_lora_fields_workflow_unchanged():
    """No LoRA fields → transform produces same nodes as baseline (no LoraLoader, no prefix)."""
    baseline_payload = _make_payload()
    result = wt.transform_app_to_vast(baseline_payload)
    wf = _workflow_from_result(result)

    lora_nodes = [n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "LoraLoader"]
    assert lora_nodes == [], "No LoraLoader expected when lora_checkpoint absent"

    pos_node = next(
        n for n in wf.values()
        if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"
        and ((n.get("_meta") or {}).get("title") or "") == _POSITIVE_TITLE
    )
    assert not (pos_node["inputs"]["text"] or "").startswith(
        ","
    ), "Positive prompt should be unchanged when lora_trigger_word absent"

    # KSampler model still references checkpoint node directly
    sampler = next(n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "KSampler")
    assert sampler["inputs"]["model"] == ["20", 0]


# ---------------------------------------------------------------------------
# Test: lora_checkpoint only → LoraLoader inserted, prompt unchanged
# ---------------------------------------------------------------------------

def test_lora_checkpoint_only_inserts_lora_node():
    """lora_checkpoint present, no trigger word → LoraLoader inserted, prompts unchanged."""
    payload = _make_payload(lora_checkpoint={"bucket": "my-bucket", "key": "loras/char.safetensors"})
    result = _run_transform(payload)
    wf = _workflow_from_result(result)

    lora_nodes = {
        nid: n for nid, n in wf.items()
        if isinstance(n, dict) and n.get("class_type") == "LoraLoader"
    }
    assert len(lora_nodes) == 1, "Exactly one LoraLoader expected"

    lora_nid, lora_node = next(iter(lora_nodes.items()))
    assert lora_node["inputs"]["strength_model"] == 1.0
    assert lora_node["inputs"]["strength_clip"] == 1.0
    assert lora_node["inputs"]["lora_name"].endswith(".safetensors")

    # KSampler model rerouted through LoraLoader
    sampler = next(n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "KSampler")
    assert sampler["inputs"]["model"] == [lora_nid, 0], "KSampler.model must come from LoraLoader"

    # Both CLIPTextEncode nodes rerouted through LoraLoader
    for node in wf.values():
        if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
            assert node["inputs"]["clip"] == [lora_nid, 1], (
                f"CLIPTextEncode clip must reference LoraLoader, got {node['inputs']['clip']}"
            )

    # Positive prompt text unchanged (no trigger word)
    pos_node = next(
        n for n in wf.values()
        if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"
        and ((n.get("_meta") or {}).get("title") or "") == _POSITIVE_TITLE
    )
    assert pos_node["inputs"]["text"] == "high quality image"


# ---------------------------------------------------------------------------
# Test: both fields present → LoraLoader inserted + trigger word prepended
# ---------------------------------------------------------------------------

def test_lora_checkpoint_and_trigger_word():
    """Both lora_checkpoint and lora_trigger_word → LoraLoader inserted + trigger word prepended."""
    trigger = "jane_doe_v1"
    payload = _make_payload(
        lora_checkpoint={"bucket": "my-bucket", "key": "loras/char.safetensors"},
        lora_trigger_word=trigger,
    )
    result = _run_transform(payload)
    wf = _workflow_from_result(result)

    lora_nodes = [n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "LoraLoader"]
    assert len(lora_nodes) == 1, "Exactly one LoraLoader expected"

    lora_nid = next(
        nid for nid, n in wf.items()
        if isinstance(n, dict) and n.get("class_type") == "LoraLoader"
    )

    # KSampler rerouted
    sampler = next(n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "KSampler")
    assert sampler["inputs"]["model"] == [lora_nid, 0]

    # Positive prompt has trigger word prepended
    pos_node = next(
        n for n in wf.values()
        if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"
        and ((n.get("_meta") or {}).get("title") or "") == _POSITIVE_TITLE
    )
    assert pos_node["inputs"]["text"].startswith(f"{trigger}, "), (
        f"Expected positive prompt to start with '{trigger}, ', got: {pos_node['inputs']['text']!r}"
    )
    assert "high quality image" in pos_node["inputs"]["text"], "Original prompt content must be preserved"

    # Negative prompt NOT modified
    neg_node = next(
        n for n in wf.values()
        if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"
        and ((n.get("_meta") or {}).get("title") or "") == _NEGATIVE_TITLE
    )
    assert not neg_node["inputs"]["text"].startswith(trigger), "Negative prompt must not have trigger word"


# ---------------------------------------------------------------------------
# Unit tests for internal helpers (no S3 needed)
# ---------------------------------------------------------------------------

def test_insert_lora_node_no_sampler_is_noop():
    """_insert_lora_node does nothing when there is no KSampler."""
    wf = {
        "30": {"inputs": {"text": "hello", "clip": ["20", 1]}, "class_type": "CLIPTextEncode", "_meta": {}},
    }
    original = copy.deepcopy(wf)
    wt._insert_lora_node(wf, "/tmp/lora.safetensors")
    assert wf == original, "Workflow must be unchanged when no KSampler found"


def test_insert_lora_node_no_clip_text_is_noop():
    """_insert_lora_node does nothing when no CLIPTextEncode with a clip ref exists."""
    wf = {
        "50": {"inputs": {"model": ["20", 0], "seed": 0}, "class_type": "KSampler", "_meta": {}},
    }
    original = copy.deepcopy(wf)
    wt._insert_lora_node(wf, "/tmp/lora.safetensors")
    assert wf == original, "Workflow must be unchanged when no CLIPTextEncode with clip ref found"


def test_prepend_trigger_word_by_title():
    """_prepend_lora_trigger_word prepends to the node matching prompt_node_title."""
    wf = {
        "30": {"inputs": {"text": "a sunset", "clip": ["20", 1]}, "class_type": "CLIPTextEncode",
               "_meta": {"title": _POSITIVE_TITLE}},
        "31": {"inputs": {"text": "blurry", "clip": ["20", 1]}, "class_type": "CLIPTextEncode",
               "_meta": {"title": _NEGATIVE_TITLE}},
    }
    wt._prepend_lora_trigger_word(wf, "char_v1", _POSITIVE_TITLE)
    assert wf["30"]["inputs"]["text"] == "char_v1, a sunset"
    assert wf["31"]["inputs"]["text"] == "blurry"


def test_prepend_trigger_word_fallback_to_positive_in_title():
    """_prepend_lora_trigger_word falls back to 'positive' in title when prompt_node_title is empty."""
    wf = {
        "30": {"inputs": {"text": "a sunset"}, "class_type": "CLIPTextEncode",
               "_meta": {"title": "Positive Prompt"}},
        "31": {"inputs": {"text": "blurry"}, "class_type": "CLIPTextEncode",
               "_meta": {"title": "Negative Prompt"}},
    }
    wt._prepend_lora_trigger_word(wf, "char_v1", "")
    assert wf["30"]["inputs"]["text"] == "char_v1, a sunset"
    assert wf["31"]["inputs"]["text"] == "blurry"


def test_new_node_id_avoids_collision():
    """_new_node_id returns an ID not already in the workflow."""
    wf = {"1": {}, "2": {}, "3": {}}
    nid = wt._new_node_id(wf)
    assert nid not in wf
    assert nid == "4"
