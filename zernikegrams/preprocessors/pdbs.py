import functools
import os
import signal
import tempfile
import stopit

import numpy as np

from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Tuple,
)
from multiprocessing import Pool

from zernikegrams.utils import log_config as logging

logger = logging.getLogger(__name__)


TIMEOUT = 300  # Max seconds per protein


@stopit.threading_timeoutable()
def process_data_dir(pdb: str, pdb_dir: str) -> Tuple[str, Tuple]:
    """
    Given a single instance of a pdb file and its parent directory,
    processes the pdb and returns a tuple of (pdb name, (*structural info))

    This function should be called in a multiprocessing routine; see
    PDBPreprocessor.execute
    """
    assert process_data_dir.callback

    pdb = pdb if isinstance(pdb, str) else pdb.decode("utf-8")
    pdb_file = os.path.join(pdb_dir, pdb + ".pdb")

    return process_data_dir.callback(pdb_file, **process_data_dir.params)


@stopit.threading_timeoutable()
def process_data_foldcomp(data: Tuple[str, str]) -> Tuple[str, Tuple]:
    """
    Given a single instance of a pdb id and pdb file contents as a string
    processes the pdb and returns a tuple of (pdb name, (*structural info))

    Param: data -- tuple of (name, pdb) where pdb is the string contents of
    a pdb file.

    This function should be called in a multiprocessing routine; see
    FoldCompPreprocessor.execute
    """
    assert process_data_foldcomp.callback

    name, pdb = data

    with tempfile.TemporaryDirectory() as temp_dir:
        with open(f"{temp_dir}/{name}.pdb", "w") as w:
            w.write(pdb)
        return process_data_foldcomp.callback(
            f"{temp_dir}/{name}.pdb", **process_data_foldcomp.params
        )


def initializer(init: Callable, init_params: Any, callback: Callable, params: Any):
    """
    Initializer function for the multiprocessing pool.

    Params:
        - init: initialization function to be called, if not None
        - init_params: params to init
        - callback: function to be called by process_data_*
        - params: parameters to callback
    """
    if init is not None:
        init(**init_params)
    process_data_dir.callback = callback
    process_data_dir.params = params
    process_data_foldcomp.callback = callback
    process_data_foldcomp.params = params
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class PDBPreprocessor:
    def __init__(self, pdb_list: List[str], pdb_dir: str):
        self.pdb_dir = pdb_dir
        self.__data = pdb_list
        self.size = len(pdb_list)
        self.pdb_name_length = np.max(list(map(len, self.__data)))

    def count(self) -> int:
        """
        Return the length of the data.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Length of the data.
        """
        return len(self.__data)

    def execute(
        self,
        callback: Callable,
        parallelism: int = 8,
        limit: int = None,
        params=None,
        init=None,
        init_params=None,
    ) -> Iterator[Tuple[str, Tuple]]:
        """
        Kicks off the multiprocessing routine for PDB files

        Params:
            - callback: function to process a single pdb
            - parallelism: max number of workers
            - limit: max number of PDBs to process
            - Params: parameters to callback
            - init: function to be called during multiprocessor pool initialization
            - init_params: parameters to init
        """
        if limit is None:
            data = self.__data
        else:
            data = self.__data[:limit]
        with Pool(
            initializer=initializer,
            processes=parallelism,
            initargs=(init, init_params, callback, params),
        ) as pool:
            all_loaded = True
            if all_loaded:
                logger.info("All PDB files are loaded.")
            else:
                msg = "Some PDB files could not be loaded."
                logger.error(msg)
                raise Exception(msg)
            process_data_pdbs = functools.partial(
                process_data_dir, pdb_dir=self.pdb_dir, timeout=TIMEOUT
            )
            for res in pool.imap(process_data_pdbs, data, chunksize=parallelism):
                if res:
                    yield res


class FoldCompPreprocessor:
    def __init__(self, pdb_list: List[str], foldcomp_file: str):
        """
        Takes the path to a foldcomp_file
        For example, data/afdb_rep_v4/afdb_rep_v4
        """
        try:
            import foldcomp
        except ModuleNotFoundError as e:
            logger.error("Foldcomp is not installed. Install with pip or bioconda")
            raise e

        self.__data = foldcomp_file
        with foldcomp.open(self.__data) as db:
            self.size = len(db)
        self.pdb_name_length = np.max(list(map(len, pdb_list)))

    def count(self) -> int:
        """
        Returns number of PDBs
        """
        return self.size

    def data(self, limit: int) -> Iterator[Tuple[str, str]]:
        """
        Generator for name, pdb tuples in FoldComp database

        Limit: integer or None
        """
        with foldcomp.open(self.__data) as db:
            for idx, (name, pdb) in enumerate(db):
                if idx >= self.count() or (limit is not None and idx >= limit):
                    break
                else:
                    yield name, pdb

    def execute(
        self,
        callback: Callable,
        parallelism: int = 8,
        limit: int = None,
        params=None,
        init=None,
        init_params=None,
    ) -> Iterator[Tuple[str, Tuple]]:
        """
        Kicks off the multiprocessing routine for PDB files

        Params:
            - callback: function to process a single pdb
            - parallelism: max number of workers
            - limit: max number of PDBs to process
            - Params: parameters to callback
            - init: function to be called during multiprocessor pool initialization
            - init_params: parameters to init
        """
        data = self.data(limit)

        with Pool(
            initializer=initializer,
            processes=parallelism,
            initargs=(init, init_params, callback, params),
        ) as pool:
            process_data_pdbs = functools.partial(
                process_data_foldcomp, timeout=TIMEOUT
            )
            for res in pool.imap(process_data_pdbs, data, chunksize=parallelism):
                if res:
                    yield res