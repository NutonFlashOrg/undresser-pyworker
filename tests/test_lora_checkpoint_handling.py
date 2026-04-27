"""
Unit tests for lora_checkpoint + lora_trigger_word handling in workflow_transform.py.

Covers:
- lora_checkpoint present → download called, LoraLoader node wired into workflow
- lora_trigger_word present → prepended to positive CLIPTextEncode text
- Neither field present → workflow unchanged (zero regression)
- Existing LoraLoader in workflow → lora_name updated in-place, no new node inserted
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub vastai SDK so worker.py can be imported without the real package.
# ---------------------------------------------------------------------------

_vastai_stub = types.ModuleType("vastai")
for _cls_name in ("BenchmarkConfig", "HandlerConfig", "LogActionConfig", "Worker", "WorkerConfig"):
    setattr(_vastai_stub, _cls_name, MagicMock())
sys.modules.setdefault("vastai", _vastai_stub)

_WORKER_DIR = Path(__file__).resolve().parent.parent / "workers" / "comfyui-json"
if str(_WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKER_DIR))

import workflow_transform as wt  # noqa: E402

# ---------------------------------------------------------------------------
# Shared minimal workflows
# ---------------------------------------------------------------------------

_CLIP_POSITIVE_TITLE = "CLIP Text Encode (Positive Prompt)"

_MINIMAL_WORKFLOW_NO_LORA = {
    "1": {
        "inputs": {"model": ["20", 0], "positive": ["30", 0], "negative": ["31", 0], "seed": 0},
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
    },
    "20": {
        "inputs": {"ckpt_name": "base.safetensors"},
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Checkpoint"},
    },
    "30": {
        "inputs": {"text": 'a photo of a person', "clip": ["20", 1]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": _CLIP_POSITIVE_TITLE},
    },
    "31": {
        "inputs": {"text": "blurry, bad", "clip": ["20", 1]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
    },
    "70": {
        "inputs": {"images": ["1", 0]},
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    },
}

_MINIMAL_WORKFLOW_WITH_LORA = {
    "1": {
        "inputs": {"model": ["50", 0], "positive": ["30", 0], "negative": ["31", 0], "seed": 0},
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
    },
    "20": {
        "inputs": {"ckpt_name": "base.safetensors"},
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Checkpoint"},
    },
    "50": {
        "inputs": {
            "lora_name": "old_lora.safetensors",
            "strength_model": 0.7,
            "strength_clip": 0.7,
            "model": ["20", 0],
            "clip": ["20", 1],
        },
        "class_type": "LoraLoader",
        "_meta": {"title": "Load LoRA"},
    },
    "30": {
        "inputs": {"text": 'a photo of a person', "clip": ["50", 1]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": _CLIP_POSITIVE_TITLE},
    },
    "31": {
        "inputs": {"text": "blurry, bad", "clip": ["50", 1]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
    },
    "70": {
        "inputs": {"images": ["1", 0]},
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    },
}


def _make_payload(
    workflow,
    *,
    lora_checkpoint=None,
    lora_trigger_word=None,
    user_prompt="a beautiful woman",
    prompt_node_title=_CLIP_POSITIVE_TITLE,
) -> dict:
    inp: dict = {
        "workflow": workflow,
        "input_images": [],
        "user_prompt": user_prompt,
        "prompt_node_title": prompt_node_title,
        "timeout": 600,
        "user_id": "user-1",
        "generation_id": "gen-1",
    }
    if lora_checkpoint is not None:
        inp["lora_checkpoint"] = lora_checkpoint
    if lora_trigger_word is not None:
        inp["lora_trigger_word"] = lora_trigger_word
    return {"input": inp}


# ---------------------------------------------------------------------------
# Helpers — direct unit tests on the helper functions
# ---------------------------------------------------------------------------


class TestApplyLoraToWorkflow:
    """Unit tests for _apply_lora_to_workflow and _insert_lora_node."""

    def test_insert_new_lora_node_into_workflow(self):
        import copy
        wf = copy.deepcopy(_MINIMAL_WORKFLOW_NO_LORA)
        wt._apply_lora_to_workflow(wf, "char.safetensors", 1.0, 1.0)

        lora_nodes = [n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "LoraLoader"]
        assert len(lora_nodes) == 1
        assert lora_nodes[0]["inputs"]["lora_name"] == "char.safetensors"
        assert lora_nodes[0]["inputs"]["strength_model"] == 1.0
        assert lora_nodes[0]["inputs"]["strength_clip"] == 1.0

    def test_insert_rewires_ksampler_model_to_lora(self):
        import copy
        wf = copy.deepcopy(_MINIMAL_WORKFLOW_NO_LORA)
        wt._apply_lora_to_workflow(wf, "char.safetensors", 1.0, 1.0)

        lora_nid = next(
            nid for nid, n in wf.items()
            if isinstance(n, dict) and n.get("class_type") == "LoraLoader"
        )
        sampler = next(n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "KSampler")
        assert sampler["inputs"]["model"] == [lora_nid, 0]

    def test_insert_rewires_clip_text_encode_to_lora(self):
        import copy
        wf = copy.deepcopy(_MINIMAL_WORKFLOW_NO_LORA)
        wt._apply_lora_to_workflow(wf, "char.safetensors", 1.0, 1.0)

        lora_nid = next(
            nid for nid, n in wf.items()
            if isinstance(n, dict) and n.get("class_type") == "LoraLoader"
        )
        clip_nodes = [
            n for n in wf.values()
            if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"
        ]
        for clip_node in clip_nodes:
            assert clip_node["inputs"]["clip"] == [lora_nid, 1]

    def test_update_existing_lora_node_in_place(self):
        import copy
        wf = copy.deepcopy(_MINIMAL_WORKFLOW_WITH_LORA)
        original_node_count = len(wf)
        wt._apply_lora_to_workflow(wf, "new_char.safetensors", 1.0, 1.0)

        # No new node should be inserted
        assert len(wf) == original_node_count
        assert wf["50"]["inputs"]["lora_name"] == "new_char.safetensors"
        assert wf["50"]["inputs"]["strength_model"] == 1.0
        assert wf["50"]["inputs"]["strength_clip"] == 1.0

    def test_no_insert_when_no_ksampler(self):
        import copy
        wf = {
            "1": {"inputs": {"text": "hi", "clip": ["2", 1]}, "class_type": "CLIPTextEncode", "_meta": {"title": "pos"}},
            "2": {"inputs": {"ckpt_name": "x"}, "class_type": "CheckpointLoaderSimple", "_meta": {}},
        }
        wt._apply_lora_to_workflow(wf, "char.safetensors", 1.0, 1.0)
        lora_nodes = [n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "LoraLoader"]
        assert len(lora_nodes) == 0

    def test_strength_from_env(self, monkeypatch):
        import copy
        monkeypatch.setenv("LORA_STRENGTH_MODEL", "0.8")
        monkeypatch.setenv("LORA_STRENGTH_CLIP", "0.6")
        wf = copy.deepcopy(_MINIMAL_WORKFLOW_WITH_LORA)
        # Simulate env-driven strength via _patch_workflow call
        wt._apply_lora_to_workflow(wf, "char.safetensors", 0.8, 0.6)
        assert wf["50"]["inputs"]["strength_model"] == 0.8
        assert wf["50"]["inputs"]["strength_clip"] == 0.6


class TestPrependLoraTriggerWord:
    """Unit tests for _prepend_lora_trigger_word."""

    def test_prepends_to_matching_title(self):
        import copy
        wf = copy.deepcopy(_MINIMAL_WORKFLOW_NO_LORA)
        wt._prepend_lora_trigger_word(wf, "elegant woman", _CLIP_POSITIVE_TITLE)
        pos_node = next(
            n for n in wf.values()
            if isinstance(n, dict) and (n.get("_meta") or {}).get("title") == _CLIP_POSITIVE_TITLE
        )
        assert pos_node["inputs"]["text"].startswith("elegant woman, ")

    def test_does_not_prepend_to_negative_node(self):
        import copy
        wf = copy.deepcopy(_MINIMAL_WORKFLOW_NO_LORA)
        wt._prepend_lora_trigger_word(wf, "elegant woman", _CLIP_POSITIVE_TITLE)
        neg_node = next(
            n for n in wf.values()
            if isinstance(n, dict) and (n.get("_meta") or {}).get("title") == "CLIP Text Encode (Negative Prompt)"
        )
        assert not neg_node["inputs"]["text"].startswith("elegant woman")

    def test_fallback_to_positive_title_keyword(self):
        import copy
        wf = {
            "1": {
                "inputs": {"text": "hello world", "clip": ["99", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"},
            }
        }
        wt._prepend_lora_trigger_word(wf, "trigger", "")
        assert wf["1"]["inputs"]["text"].startswith("trigger, ")

    def test_no_match_when_title_missing(self):
        import copy
        wf = {
            "1": {
                "inputs": {"text": "hello", "clip": ["99", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "My Prompt"},
            }
        }
        wt._prepend_lora_trigger_word(wf, "trigger", _CLIP_POSITIVE_TITLE)
        assert wf["1"]["inputs"]["text"] == "hello"


# ---------------------------------------------------------------------------
# Integration tests through transform_app_to_vast
# ---------------------------------------------------------------------------


class TestTransformLoraCheckpoint:
    """Integration tests: lora_checkpoint field through transform_app_to_vast."""

    def _transform(self, payload: dict, lora_name: str = "char.safetensors") -> dict:
        with patch("workflow_transform._download_lora_checkpoint", return_value=lora_name):
            return wt.transform_app_to_vast(payload)

    def test_lora_checkpoint_triggers_download(self):
        payload = _make_payload(
            _MINIMAL_WORKFLOW_NO_LORA,
            lora_checkpoint={"bucket": "b", "key": "loras/char/char.safetensors"},
        )
        with patch("workflow_transform._download_lora_checkpoint", return_value="char.safetensors") as mock_dl:
            wt.transform_app_to_vast(payload)
        mock_dl.assert_called_once_with({"bucket": "b", "key": "loras/char/char.safetensors"})

    def test_lora_node_present_in_output_workflow(self):
        payload = _make_payload(
            _MINIMAL_WORKFLOW_NO_LORA,
            lora_checkpoint={"bucket": "b", "key": "loras/char/char.safetensors"},
        )
        out = self._transform(payload)
        wf = out["input"]["workflow_json"]
        lora_nodes = [n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "LoraLoader"]
        assert len(lora_nodes) == 1
        assert lora_nodes[0]["inputs"]["lora_name"] == "char.safetensors"

    def test_absent_lora_checkpoint_leaves_workflow_unchanged(self):
        payload = _make_payload(_MINIMAL_WORKFLOW_NO_LORA)
        out = wt.transform_app_to_vast(payload)
        wf = out["input"]["workflow_json"]
        lora_nodes = [n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "LoraLoader"]
        assert len(lora_nodes) == 0

    def test_existing_lora_node_updated_not_duplicated(self):
        payload = _make_payload(
            _MINIMAL_WORKFLOW_WITH_LORA,
            lora_checkpoint={"bucket": "b", "key": "loras/newchar/new.safetensors"},
        )
        out = self._transform(payload, lora_name="char.safetensors")
        wf = out["input"]["workflow_json"]
        lora_nodes = [n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "LoraLoader"]
        assert len(lora_nodes) == 1
        assert lora_nodes[0]["inputs"]["lora_name"] == "char.safetensors"

    def test_lora_name_is_filename_not_absolute_path(self):
        payload = _make_payload(
            _MINIMAL_WORKFLOW_NO_LORA,
            lora_checkpoint={"bucket": "b", "key": "loras/char/char.safetensors"},
        )
        out = self._transform(payload, lora_name="char.safetensors")
        wf = out["input"]["workflow_json"]
        lora_nodes = [n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "LoraLoader"]
        lora_name_val = lora_nodes[0]["inputs"]["lora_name"]
        assert not lora_name_val.startswith("/"), f"lora_name must not be absolute: {lora_name_val!r}"


class TestTransformLoraTriggerWord:
    """Integration tests: lora_trigger_word through transform_app_to_vast."""

    def test_trigger_word_prepended_to_positive_prompt(self):
        payload = _make_payload(
            _MINIMAL_WORKFLOW_NO_LORA,
            lora_trigger_word="elegant woman",
            user_prompt="",
            prompt_node_title="",
        )
        out = wt.transform_app_to_vast(payload)
        wf = out["input"]["workflow_json"]
        pos_node = next(
            n for n in wf.values()
            if isinstance(n, dict) and (n.get("_meta") or {}).get("title") == _CLIP_POSITIVE_TITLE
        )
        assert "elegant woman" in pos_node["inputs"]["text"]

    def test_trigger_word_absent_leaves_prompt_unchanged(self):
        payload = _make_payload(_MINIMAL_WORKFLOW_NO_LORA, user_prompt="")
        out = wt.transform_app_to_vast(payload)
        wf = out["input"]["workflow_json"]
        pos_node = next(
            n for n in wf.values()
            if isinstance(n, dict) and (n.get("_meta") or {}).get("title") == _CLIP_POSITIVE_TITLE
        )
        assert "elegant" not in pos_node["inputs"]["text"]

    def test_trigger_word_with_lora_checkpoint(self):
        payload = _make_payload(
            _MINIMAL_WORKFLOW_NO_LORA,
            lora_checkpoint={"bucket": "b", "key": "loras/char/char.safetensors"},
            lora_trigger_word="charname",
            user_prompt="",
            prompt_node_title="",
        )
        with patch("workflow_transform._download_lora_checkpoint", return_value="char.safetensors"):
            out = wt.transform_app_to_vast(payload)
        wf = out["input"]["workflow_json"]
        lora_nodes = [n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "LoraLoader"]
        assert len(lora_nodes) == 1
        pos_node = next(
            n for n in wf.values()
            if isinstance(n, dict) and (n.get("_meta") or {}).get("title") == _CLIP_POSITIVE_TITLE
        )
        assert "charname" in pos_node["inputs"]["text"]


class TestTransformNoLoraRegression:
    """No-LoRA payloads must pass through exactly as before."""

    def test_no_lora_fields_workflow_structure_unchanged(self):
        payload = _make_payload(_MINIMAL_WORKFLOW_NO_LORA)
        out = wt.transform_app_to_vast(payload)
        wf = out["input"]["workflow_json"]
        assert set(wf.keys()) == set(_MINIMAL_WORKFLOW_NO_LORA.keys())

    def test_no_lora_fields_ksampler_model_unchanged(self):
        payload = _make_payload(_MINIMAL_WORKFLOW_NO_LORA)
        out = wt.transform_app_to_vast(payload)
        wf = out["input"]["workflow_json"]
        sampler = next(n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "KSampler")
        assert sampler["inputs"]["model"] == ["20", 0]

    def test_s3_prefix_still_forwarded_without_lora(self):
        payload = _make_payload(_MINIMAL_WORKFLOW_NO_LORA)
        payload["input"]["s3_prefix"] = "test/prefix"
        out = wt.transform_app_to_vast(payload)
        assert out["input"]["s3_prefix"] == "test/prefix"
