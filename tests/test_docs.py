from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_names_the_new_checkpoint_and_runtime() -> None:
    text = README.read_text(encoding="utf-8")
    assert "nvidia/GLM-5.3-Flash-FP4-Local" in text
    assert "eugr/spark-vllm-b12x:latest" in text
    assert "https://github.com/eugr/spark-vllm-docker" in text
    assert "2,000,000" in text
    assert "17G" in text
    assert "2,078,358" in text
    assert "127.0.0.1" in text
    assert "./launch.sh" in text


def test_readme_does_not_advertise_the_old_weights() -> None:
    text = README.read_text(encoding="utf-8")
    for stale in (
        "brandonmusic/GLM-5.3-Flash-tr3-4bpw",
        "incoai/GLM-5.3-Flash-DFlash2",
        "glm53-selfbuild",
        "Dockerfile.e3",
    ):
        assert stale not in text, stale


def test_docs_exist_and_cover_the_required_topics() -> None:
    expected = {
        "01-architecture.md": ("KV cache", "45 layers", "TP=2"),
        "02-parameters.md": ("kv-cache-memory-bytes", "modelopt_mixed", "B12X"),
        "03-bringup.md": ("preflight", "launch.sh setup", "smoke"),
        "04-kv-cache.md": ("8,800", "2,000,000", "17 GiB"),
        "05-troubleshooting.md": ("modelopt_mixed", "NCCL", "BELOW TARGET"),
        "06-upstream.md": ("0bdc9c94", "03a956", "MIT"),
    }
    for name, needles in expected.items():
        path = ROOT / "docs" / name
        assert path.is_file(), name
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, f"{name}: {needle}"


def test_speed_receipts_are_present() -> None:
    import json

    matrix = ROOT / "docs" / "receipts" / "2026-10-17-speed-matrix.json"
    repeat = ROOT / "docs" / "receipts" / "2026-10-17-speed-matrix-decode-rep2.json"
    assert matrix.is_file() and repeat.is_file()
    data = json.loads(matrix.read_text(encoding="utf-8"))
    assert data["ok"] is True
    assert len(data["cells"]) == 24
    decode = [cell for cell in data["cells"] if cell["kind"] == "decode"]
    assert [cell["concurrency"] for cell in decode] == [1, 2, 4, 8]
    assert all(cell["completion_tokens_median"] == 200 for cell in decode)


def test_status_records_the_replacement() -> None:
    text = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    assert "GLM-5.3-Flash-FP4-Local" in text
    assert "EXL3" in text  # the removal is the change


def test_notice_names_upstream_and_model_license() -> None:
    text = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "spark-vllm-docker" in text
    assert "NVIDIA Open Model Agreement" in text
    assert "MIT License" in text
