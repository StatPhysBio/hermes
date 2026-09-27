"""
Tests for the Wigner 3j coefficient cache.

The cache used to be written straight to its destination inside the installed
package, which had three consequences:

  - a run interrupted mid-write left a truncated file that made every later run
    fail with `EOFError: Ran out of input`, permanently, until it was removed by
    hand;
  - the coefficients could not be cached at all on a read-only installation;
  - every new environment recomputed them from scratch.

All tests here use a small `lmax`, so they take seconds rather than the minutes
the real `lmax=14` table needs. They need neither model weights nor pyrosetta.
"""

import glob
import gzip
import os
import sys
import tempfile

import pytest

# Exercise the source tree these tests ship with, not whatever happens to be
# installed in site-packages.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes.cg_coefficients import get_w3j_coefficients
from hermes.cg_coefficients.get_w3j_coefficients import (
    _cache_dirs,
    _cache_filename,
    _write_cache,
    curr_file_path,
)

# `hermes.cg_coefficients.__init__` rebinds this name to the function, so the
# module itself has to come out of sys.modules.
w3j = sys.modules["hermes.cg_coefficients.get_w3j_coefficients"]

LMAX = 2


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """
    Confine the cache to one private directory.

    `_cache_dirs()` is replaced wholesale rather than steered with environment
    variables: the real chain also includes the user cache directory and the
    package directory, so without this the tests would read each other's
    leftovers and would write into the working copy.
    """
    d = tmp_path / "cache"
    d.mkdir()
    monkeypatch.setattr(w3j, "_cache_dirs", lambda: [str(d)])
    return d


# --------------------------------------------------------------- caching itself

def test_coefficients_are_cached_and_reused(cache_dir):
    first = get_w3j_coefficients(lmax=LMAX)
    assert (cache_dir / _cache_filename(LMAX)).exists(), f"nothing written to {cache_dir}"

    # The second call must come off disk and agree with the first.
    second = get_w3j_coefficients(lmax=LMAX)
    assert first.keys() == second.keys()
    assert all((first[k] == second[k]).all() for k in first)


def test_cached_file_is_valid_gzip(cache_dir):
    get_w3j_coefficients(lmax=LMAX)
    with gzip.open(cache_dir / _cache_filename(LMAX), "rb") as f:
        assert f.read(1), "the cached file is empty"


# ----------------------------------------------------------- corrupt cache files

def test_truncated_cache_is_recovered(cache_dir):
    """A 0-byte file used to poison every subsequent run, forever."""
    poisoned = cache_dir / _cache_filename(LMAX)
    poisoned.write_bytes(b"")

    with pytest.warns(UserWarning, match="unreadable Wigner 3j cache"):
        coefficients = get_w3j_coefficients(lmax=LMAX)

    assert len(coefficients) > 0
    assert poisoned.stat().st_size > 0, "the poisoned cache was not replaced"


def test_garbled_cache_is_recovered(cache_dir):
    """Same, for a file that is not even valid gzip."""
    garbled = cache_dir / _cache_filename(LMAX)
    garbled.write_bytes(b"this is not a gzip file")

    with pytest.warns(UserWarning, match="unreadable Wigner 3j cache"):
        assert len(get_w3j_coefficients(lmax=LMAX)) > 0

    with gzip.open(garbled, "rb") as f:
        assert f.read(1), "the garbled cache was not replaced with a valid one"


def test_corrupt_cache_is_removed_even_when_another_copy_is_found(tmp_path, monkeypatch):
    """
    With a valid copy in a later directory there is nothing to recompute, but
    the corrupt file must still go, or every later run repeats the failed read.
    """
    bad, good = tmp_path / "bad", tmp_path / "good"
    bad.mkdir(), good.mkdir()
    monkeypatch.setattr(w3j, "_cache_dirs", lambda: [str(bad), str(good)])

    _write_cache(str(good / _cache_filename(LMAX)), {"sentinel": 1})
    poisoned = bad / _cache_filename(LMAX)
    poisoned.write_bytes(b"")

    with pytest.warns(UserWarning, match="unreadable Wigner 3j cache"):
        assert get_w3j_coefficients(lmax=LMAX) == {"sentinel": 1}

    assert not poisoned.exists(), "the corrupt cache file was left behind"


# ------------------------------------------------------------------ atomic write

def test_failed_write_leaves_nothing_behind(cache_dir):
    """A write that blows up must not leave a partial or temporary file."""
    path = str(cache_dir / _cache_filename(LMAX))

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _write_cache(path, {"bad": Unpicklable()})

    assert not os.path.exists(path), "a partial cache file was left at the destination"
    assert not glob.glob(str(cache_dir / "*.tmp")), "a temporary file was left behind"


def test_write_replaces_an_existing_cache_without_truncating_it_first(cache_dir):
    path = str(cache_dir / _cache_filename(LMAX))
    _write_cache(path, {"old": 1})
    _write_cache(path, {"new": 2})
    with gzip.open(path, "rb") as f:
        import pickle
        assert pickle.load(f) == {"new": 2}


def test_unwritable_directory_falls_back_to_the_temp_dir(tmp_path, monkeypatch):
    """A read-only installation must still cache, just somewhere else."""
    read_only = tmp_path / "read_only"
    read_only.mkdir()
    read_only.chmod(0o500)
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(w3j, "_TEMP_CACHE_DIR", str(fallback))
    monkeypatch.setattr(w3j, "_cache_dirs", lambda: [str(read_only), str(fallback)])
    try:
        assert len(get_w3j_coefficients(lmax=LMAX)) > 0
    finally:
        read_only.chmod(0o700)

    assert (fallback / _cache_filename(LMAX)).exists(), "did not fall back to the temp dir"
    assert not list(read_only.glob("*")), "wrote into the read-only directory"


def test_warns_when_nothing_at_all_is_writable(tmp_path, monkeypatch):
    """Still usable, but the user is told they will pay for it every run."""
    read_only = tmp_path / "read_only"
    read_only.mkdir()
    read_only.chmod(0o500)
    unwritable_temp = read_only / "hermes-cache"  # makedirs under it will fail
    monkeypatch.setattr(w3j, "_TEMP_CACHE_DIR", str(unwritable_temp))
    monkeypatch.setattr(w3j, "_cache_dirs", lambda: [str(read_only), str(unwritable_temp)])
    try:
        with pytest.warns(UserWarning, match="[Cc]ould not cache"):
            assert len(get_w3j_coefficients(lmax=LMAX)) > 0
    finally:
        read_only.chmod(0o700)


# ------------------------------------------------------- which directories, in order

def test_no_directory_named_tilde_when_home_is_unset(tmp_path, monkeypatch):
    """
    `os.path.expanduser('~')` returns a literal '~' when HOME is unset and the
    passwd lookup fails, which would otherwise create a directory *named* '~'
    in the working directory - never to be found again.
    """
    monkeypatch.delenv("HERMES_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(os.path, "expanduser", lambda path: path)
    monkeypatch.chdir(tmp_path)

    dirs = _cache_dirs()
    assert not any("~" in d.split(os.sep) for d in dirs), dirs
    assert not (tmp_path / "~").exists()


def test_explicit_override_is_searched_first(monkeypatch):
    monkeypatch.setenv("HERMES_CACHE_DIR", "/somewhere/explicit")
    assert _cache_dirs()[0] == os.path.abspath("/somewhere/explicit")


def test_package_directory_is_still_searched():
    """Existing installations must keep finding the copy they already computed."""
    assert os.path.abspath(curr_file_path) in _cache_dirs()


def test_temp_directory_is_the_last_resort():
    assert _cache_dirs()[-1] == os.path.abspath(
        os.path.join(tempfile.gettempdir(), "hermes-cache")
    )


def test_duplicate_directories_are_collapsed(monkeypatch):
    monkeypatch.setenv("HERMES_CACHE_DIR", curr_file_path)
    dirs = _cache_dirs()
    assert len(dirs) == len(set(dirs)), dirs
