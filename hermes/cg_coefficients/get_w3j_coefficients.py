import os
import gzip, pickle
import tempfile
import warnings
import zlib
import e3nn
from e3nn import o3

curr_file_path = os.path.dirname(os.path.abspath(__file__))

# Errors that mean "this file is truncated or garbled", as opposed to a
# transient filesystem problem. Only these make us fall through to the next
# candidate and, in the end, recompute the coefficients: a flaky network mount
# should raise loudly rather than silently trigger a multi-minute recomputation.
_CORRUPT_CACHE_ERRORS = (EOFError, gzip.BadGzipFile, zlib.error, pickle.UnpicklingError)

# Always available, on every operating system, and honours TMPDIR/TEMP/TMP.
# Only used when nothing else is writable, since it does not survive a reboot.
_TEMP_CACHE_DIR = os.path.join(tempfile.gettempdir(), 'hermes-cache')


def _cache_filename(lmax):
    return 'w3j_matrices-lmax=%d-version=%s.gz' % (lmax, e3nn.__version__)


def _cache_dirs():
    '''
    Every directory we are willing to keep the coefficients in, best first.

    The table takes minutes to compute but only a couple of MB to store, so we
    read from the first location that has it and write to every location that
    accepts it. That way a fresh conda environment, a job that runs with a
    different HOME, and a read-only installation all find an existing copy
    instead of each recomputing their own.
    '''
    dirs = []

    # 1. Explicit override. The only location guaranteed to work on any system,
    #    and the escape hatch when the ones below are unsuitable.
    if os.environ.get('HERMES_CACHE_DIR'):
        dirs.append(os.environ['HERMES_CACHE_DIR'])

    # 2. The user cache directory. `expanduser` returns a literal '~' when HOME
    #    is unset and the passwd lookup fails (containers, some batch jobs);
    #    without this guard we would create a directory *named* '~' in the
    #    working directory and never find it again.
    home = os.path.expanduser('~')
    if home != '~' and os.path.isdir(home):
        xdg = os.environ.get('XDG_CACHE_HOME') or os.path.join(home, '.cache')
        dirs.append(os.path.join(xdg, 'hermes'))

    # 3. Next to this file. Where hermes has always kept it, so that existing
    #    installations keep using the copy they have already computed.
    dirs.append(curr_file_path)

    # 4. Last resort, so that a read-only home and a read-only installation
    #    still work.
    dirs.append(_TEMP_CACHE_DIR)

    # A user who points HERMES_CACHE_DIR at one of the others should not make
    # us write the same file twice.
    return list(dict.fromkeys(os.path.abspath(d) for d in dirs))


def _read_cache(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def _write_cache(path, w3j_matrices):
    '''
    Write the coefficients atomically.

    An interrupted or concurrent run must never leave a partial file at `path`:
    the old implementation opened the destination directly, so a run killed
    during the several minutes it takes to fill left a truncated file behind,
    and every later run then failed with `EOFError: Ran out of input` until it
    was deleted by hand.
    '''
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    # The temporary file has to live in the destination directory: os.replace
    # is atomic, but it cannot rename across filesystems.
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix='.tmp')
    os.close(fd)
    try:
        with gzip.open(tmp_path, 'wb') as f:
            pickle.dump(w3j_matrices, f)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def compute_w3j_coefficients(lmax=14):
    '''Compute the Wigner 3j matrices up to `lmax`, without touching any cache.'''
    w3j_matrices = {}
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            for l3 in range(abs(l2 - l1), min(l2 + l1, lmax) + 1):
                w3j_matrices[(l1, l2, l3)] = o3.wigner_3j(l1, l2, l3).numpy()
    return w3j_matrices


def download_w3j_coefficients(lmax=14):
    '''
    Compute the Wigner 3j coefficients and cache them wherever possible.

    Returns `(w3j_matrices, paths_written)`. Nothing is actually downloaded -
    the coefficients are computed locally with e3nn - but the name is kept for
    backwards compatibility.
    '''
    print('e3nn version: ', e3nn.__version__)
    w3j_matrices = compute_w3j_coefficients(lmax=lmax)

    written = []
    for directory in _cache_dirs():
        if directory == os.path.abspath(_TEMP_CACHE_DIR):
            continue  # only as a fallback, handled below
        try:
            path = os.path.join(directory, _cache_filename(lmax))
            _write_cache(path, w3j_matrices)
            written.append(path)
        except OSError:
            continue  # read-only, out of quota, ... just try the next one

    if not written:
        try:
            path = os.path.join(_TEMP_CACHE_DIR, _cache_filename(lmax))
            _write_cache(path, w3j_matrices)
            written.append(path)
        except OSError:
            warnings.warn(
                'Could not cache the Wigner 3j coefficients in any of %s, so they '
                'will be recomputed on every run. Set HERMES_CACHE_DIR to a '
                'writable directory to avoid this.' % (_cache_dirs(),)
            )

    return w3j_matrices, written


def get_w3j_coefficients(lmax=14):
    filename = _cache_filename(lmax)

    for directory in _cache_dirs():
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            continue
        try:
            return _read_cache(path)
        except _CORRUPT_CACHE_ERRORS as e:
            warnings.warn(
                'Ignoring unreadable Wigner 3j cache %s (%s). It was most likely '
                'left behind by an interrupted run; removing it.' % (path, e)
            )
            # A corrupt file has no value, and leaving it in place would make
            # every later run pay for the same failed read.
            try:
                os.remove(path)
            except OSError:
                pass

    print('Computing Wigner 3j coefficients (lmax=%d). This takes a few minutes, '
          'and is cached for future runs.' % (lmax,))
    w3j_matrices, written = download_w3j_coefficients(lmax=lmax)
    if written:
        print('Cached Wigner 3j coefficients in: %s' % (', '.join(written),))
    return w3j_matrices
