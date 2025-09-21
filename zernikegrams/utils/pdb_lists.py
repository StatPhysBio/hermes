import os

from typing import List


def pdb_list_from_dir(pdb_dir: str) -> List[str]:
    """
    Create a list of pdb files from a give directory
    """
    x = [f.removesuffix(".pdb") for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    return x


def pdb_list_from_foldcomp(foldcomp: str) -> List[str]:
    """
    Create a list of pdb files from a give foldcomp db file
    """
    file = f"{foldcomp}.lookup"
    lines = [line.split() for line in open(file).readlines()]
    return [ID for (_, ID, _) in lines]