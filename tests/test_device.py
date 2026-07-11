"""Contracts for GPU-first model device selection and MinerU propagation."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from types import SimpleNamespace

import pytest

from ingest import parse_mineru


def load_device_module():
    assert importlib.util.find_spec("runtime") is not None, "runtime should exist"
    assert importlib.util.find_spec("runtime.device") is not None, (
        "runtime.device should exist"
    )
    return importlib.import_module("runtime.device")


def fake_torch(*, cuda_available: bool):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        float16=object(),
    )


def test_auto_prefers_cuda_without_warning(monkeypatch) -> None:
    device = load_device_module()
    monkeypatch.delenv("MODEL_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda_available=True))

    with warnings.catch_warnings(record=True) as warnings_seen:
        warnings.simplefilter("always")
        selected = device.select_model_device("bge-m3")

    assert selected == "cuda"
    assert len(warnings_seen) == 0


def test_auto_warns_when_falling_back_to_cpu(monkeypatch) -> None:
    device = load_device_module()
    monkeypatch.delenv("MODEL_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda_available=False))

    with pytest.warns(RuntimeWarning, match="bge-m3.*falling back to CPU"):
        assert device.select_model_device("bge-m3") == "cpu"


def test_explicit_cuda_warns_and_falls_back_when_unavailable(monkeypatch) -> None:
    device = load_device_module()
    monkeypatch.setenv("MODEL_DEVICE", "cuda")
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda_available=False))

    with pytest.warns(RuntimeWarning, match="mineru.*falling back to CPU"):
        assert device.select_model_device("mineru") == "cpu"


def test_explicit_cpu_warns(monkeypatch) -> None:
    device = load_device_module()
    monkeypatch.setenv("MODEL_DEVICE", "cpu")

    with pytest.warns(RuntimeWarning, match="reranker.*explicitly configured"):
        assert device.select_model_device("reranker") == "cpu"


def test_invalid_model_device_fails(monkeypatch) -> None:
    device = load_device_module()
    monkeypatch.setenv("MODEL_DEVICE", "tpu")

    with pytest.raises(ValueError, match="MODEL_DEVICE"):
        device.select_model_device("bge-m3")


def test_model_kwargs_use_fp16_only_on_cuda(monkeypatch) -> None:
    device = load_device_module()
    torch = fake_torch(cuda_available=True)
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert device.model_kwargs_for("cuda") == {"dtype": torch.float16}
    assert device.model_kwargs_for("cpu") is None


def test_mineru_subprocess_receives_selected_cuda_device(
    monkeypatch, tmp_path
) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"pdf fixture")
    work = tmp_path / "mineru"
    produced = work / pdf.stem / "ocr" / f"{pdf.stem}.md"
    produced.parent.mkdir(parents=True)
    produced.write_text("parsed", encoding="utf-8")
    captured = {}

    def fake_run(command, *, check, env):
        captured.update(command=command, check=check, env=env)

    monkeypatch.setattr(parse_mineru, "WORK", work)
    monkeypatch.setattr(parse_mineru, "select_model_device", lambda _name: "cuda", raising=False)
    monkeypatch.setattr(parse_mineru.subprocess, "run", fake_run)

    assert parse_mineru.run_mineru(pdf) == produced
    assert captured["check"] is True
    assert captured["env"]["MINERU_MODEL_SOURCE"] == "modelscope"
    assert captured["env"]["MINERU_DEVICE_MODE"] == "cuda"


def test_mineru_subprocess_receives_selected_cpu_device(monkeypatch, tmp_path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"pdf fixture")
    work = tmp_path / "mineru"
    produced = work / pdf.stem / "ocr" / f"{pdf.stem}.md"
    produced.parent.mkdir(parents=True)
    produced.write_text("parsed", encoding="utf-8")
    captured = {}

    def fake_run(command, *, check, env):
        captured.update(command=command, check=check, env=env)

    monkeypatch.setattr(parse_mineru, "WORK", work)
    monkeypatch.setattr(parse_mineru, "select_model_device", lambda _name: "cpu")
    monkeypatch.setattr(parse_mineru.subprocess, "run", fake_run)

    assert parse_mineru.run_mineru(pdf) == produced
    assert captured["env"]["MINERU_DEVICE_MODE"] == "cpu"
