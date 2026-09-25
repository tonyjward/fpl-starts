"""`data/` (local-only historical inputs) and `models/` (locally fitted
artefacts) must never become part of a normal commit."""

import os
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not os.path.isdir(os.path.join(REPO_ROOT, ".git")),
    reason="needs a git checkout",
)


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


@pytest.mark.parametrize("path", [
    "data/availability/historical_availability.csv",
    "data/vaastav/2025-26/gws/merged_gw.csv",
    "models/logistic_availability/v1/model.json",
])
def test_private_and_artefact_paths_are_gitignored(path):
    assert _git("check-ignore", "-q", path).returncode == 0, path


def test_nothing_under_data_or_models_is_tracked():
    tracked = _git("ls-files", "data", "models").stdout.split()
    assert tracked == []
