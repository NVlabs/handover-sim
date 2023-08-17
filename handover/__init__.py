# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

"""Handover-Sim package."""


# NOTE (roflaherty): This is inspired by how matplotlib does creates its version value.
# https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/__init__.py#L161
def _get_version() -> str:
    """Return the version string used for __version__."""
    # Standard Library
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    if (root / ".git").exists() and not (root / ".git/shallow").exists():
        # Third Party
        import setuptools_scm

        this_version: str
        # See the `setuptools_scm` documentation for the description of the schemes used below.
        # https://pypi.org/project/setuptools-scm/
        # NOTE: If these values are updated, they need to be also updated in `pyproject.toml`.
        this_version = setuptools_scm.get_version(
            root=root,
            version_scheme="no-guess-dev",
            local_scheme="dirty-tag",
        )
    else:  # Get the version from the _version.py setuptools_scm file.
        try:
            # Standard Library
            from importlib.metadata import version
        except ModuleNotFoundError:
            # NOTE: `importlib.resources` is part of the standard library in Python 3.9.
            # `importlib_metadata` is the back ported library for older versions of python.
            # Third Party
            from importlib_metadata import version  # type: ignore[no-redef]

        this_version = version("handover-sim")

    return this_version


# Set `__version__` attribute
__version__ = _get_version()

# Remove `_get_version` so it is not added as an attribute
del _get_version

from gym.envs.registration import register

register(
    id="HandoverStateEnv-v1",
    entry_point="handover.handover_env:HandoverStateEnv",
)

register(
    id="HandoverHandCameraPointStateEnv-v1",
    entry_point="handover.handover_env:HandoverHandCameraPointStateEnv",
)
