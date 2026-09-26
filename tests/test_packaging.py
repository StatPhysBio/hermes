"""
Regression tests for the packaging of the non-python data files.

`zernikegrams.structural_info.structural_info_core` and
`zernikegrams.holograms.get_holograms` open `charges.rtp` and
`YZX_XYZ_cob.npy` at import time, relative to their own `__file__`.  If those
two files are not copied into site-packages by `pip install .`, importing
`hermes`/`zernikegrams` from outside the repository dies with a
FileNotFoundError.  Inside the repository the bug is invisible, because the
source tree shadows the installed package.

These tests are cheap and require neither pyrosetta nor a trained model.
"""

import os
import subprocess
import sys
import zipfile
from importlib import import_module

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (module that loads the file, name of the data file, path inside the wheel)
DATA_FILES = [
    (
        "zernikegrams.structural_info.structural_info_core",
        "charges.rtp",
        "zernikegrams/structural_info/charges.rtp",
    ),
    (
        "zernikegrams.holograms.get_holograms",
        "YZX_XYZ_cob.npy",
        "zernikegrams/holograms/YZX_XYZ_cob.npy",
    ),
]


@pytest.mark.parametrize("module_name, filename, _", DATA_FILES)
def test_data_file_sits_next_to_the_module_that_loads_it(module_name, filename, _):
    """The data file must be found wherever `zernikegrams` is imported from."""
    module = import_module(module_name)
    expected = os.path.join(os.path.dirname(os.path.abspath(module.__file__)), filename)
    assert os.path.exists(expected), (
        f"{filename} is missing from {os.path.dirname(expected)}. "
        f"If this is an installed copy of the package, it means the file was "
        f"not declared as package_data in setup.py."
    )


def test_wheel_contains_the_data_files(tmp_path):
    """`pip install .` must ship the data files inside the wheel.

    This is the check that actually catches the bug: the tests above pass
    trivially when run from the repository, since the source tree shadows the
    installed package.
    """
    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "wheel",
            "--no-deps", "--no-build-isolation",
            "--wheel-dir", str(tmp_path),
            REPO_ROOT,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"could not build the wheel in this environment:\n{result.stderr}")

    wheels = list(tmp_path.glob("hermes-*.whl"))
    assert len(wheels) == 1, f"expected exactly one hermes wheel, got {wheels}"

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())

    for _, filename, path_in_wheel in DATA_FILES:
        assert path_in_wheel in names, (
            f"{filename} is not in the wheel built from {REPO_ROOT}; "
            f"an installation made from it would fail on `import hermes`."
        )
