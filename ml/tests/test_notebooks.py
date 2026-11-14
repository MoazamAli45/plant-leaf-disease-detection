"""Structure checks for the generated Kaggle notebook (it is run unattended for hours)."""

import ast
import importlib.util
from pathlib import Path

import nbformat
import pytest

NB_DIR = Path(__file__).parents[1] / "notebooks"


def load_builder():
    spec = importlib.util.spec_from_file_location("b", NB_DIR / "build_kaggle_phase3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_notebook_matches_builder():
    built = load_builder().build()
    on_disk = nbformat.read(NB_DIR / "kaggle_phase3.ipynb", as_version=4)
    assert [(c.cell_type, c.source) for c in on_disk.cells] == [
        (c.cell_type, c.source) for c in built.cells
    ], "run: python notebooks/build_kaggle_phase3.py"


def test_cell_order_and_types():
    cells = load_builder().build().cells
    assert cells[0].cell_type == "markdown" and cells[0].source.startswith("# Phase 3")
    headers = [c.source.splitlines()[0] for c in cells[1:]]
    assert [c.cell_type for c in cells[1:]] == ["code"] * 6
    for key, h in zip(
        ["settings", "code, NPD", "GPU queues", "smoke pass", "real runs", "pack"],
        headers,
        strict=True,
    ):
        assert key in h, h


@pytest.mark.parametrize("i", range(1, 7))
def test_code_cells_are_valid_python(i):
    ast.parse(load_builder().build().cells[i].source)


def test_batches_reference_existing_configs():
    ns: dict = {}
    exec(load_builder().SETTINGS, ns)  # settings cell has no side effects beyond printing
    root = Path(__file__).parents[1]
    for configs in ns["BATCHES"].values():
        for c in configs:
            assert (root / c).exists(), c
    assert ns["TAG"] == "main_seed42"
