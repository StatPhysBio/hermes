import setuptools
from setuptools.command.install import install
import os
import subprocess


class CustomInstall(install):
    def run(self):
        install.run(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))

        ## add stuff here if necessary



setuptools.setup(
    name='hermes',
    version='0.1.0',
    author='Gian Marco Visani',
    author_email='gvisan01@.cs.washington.edu',
    description='Learning protein neighborhoods by incorporating rotational symmetry - web version',
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/StatPhysBio/hermes',
    python_requires='>=3.9',
    packages=setuptools.find_packages(),
    include_package_data=True,
    cmdclass={"install": CustomInstall},
    install_requires=[
        "argparse",
        "cmake",
        "foldcomp",
        "biopython==1.83",
        "h5py==3.10.0",
        "hdf5plugin==4.4.0",
        "numpy==1.24.3",
        "matplotlib",
        "e3nn==0.5.0",
        "pandas==1.5.0",
        "tqdm"
    ]
)
