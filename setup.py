# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

"""Handover-Sim package setuptools."""

# NOTE (roflaherty): This file is still needed to allow the package to be
# installed in editable mode.
#
# References:
# * https://setuptools.pypa.io/en/latest/setuptools.html#setup-cfg-only-projects

# Standard Library
import subprocess
import sys

# Third Party
import setuptools
from setuptools.command.develop import develop as _develop


class develop(_develop):
    def run(self):
        _develop.run(self)

        # Ideally, one should be able to also install the dependent Python packages resided in
        # submodules with one pip install run from the main repo. This is possible using
        # `install_requires` with file URLs:
        # * https://stackoverflow.com/questions/28113862/how-to-install-a-dependency-from-a-submodule-in-python
        # * https://stackoverflow.com/questions/64878322/setuptools-find-package-and-automatic-submodule-dependancies-management
        # * https://stackoverflow.com/questions/64988110/using-git-submodules-with-python
        # * https://www.python.org/dev/peps/pep-0440/#direct-references
        #
        # For example:
        #
        #   install_requires=[
        #       f'mano_pybullet @ file://{os.getcwd()}/mano_pybullet',
        #   ]
        #
        # However, this does not work for `setup.cfg` currently since pip does not support non-local
        # file URLs.
        # * https://github.com/pypa/pip/issues/6658#issuecomment-506841157
        #
        # Besides, it is currently not possible to use this method for editable (develop) mode.
        # * https://stackoverflow.com/questions/68491950/specifying-develop-mode-for-setuptools-setup-install-requires-argument

        # Use the subprocess hack. This also allows running with `--no-deps`.
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--verbose",
                "--no-cache-dir",
                "--no-deps",
                "--editable",
                "mano_pybullet",
            ]
        )


setuptools.setup(
    cmdclass={"develop": develop},
)
