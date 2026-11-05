"""Serving tests hold FAISS and ONNX Runtime and never import PyTorch in this process."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

BUNDLE_ENV = "PATCHSENTRY_TEST_BUNDLE"
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real, small bundle: an untrained ResNet-18 exported to ONNX, plus a fitted memory bank."""
    prebuilt = os.environ.get(BUNDLE_ENV)
    if prebuilt:
        return Path(prebuilt)
    directory = tmp_path_factory.mktemp("bundle")
    environment = {**os.environ, "PYTHONPATH": f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}"}
    result = subprocess.run(
        [sys.executable, "-m", "tests.make_bundle", str(directory)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return directory
