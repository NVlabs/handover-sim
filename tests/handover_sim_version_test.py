# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

"""Unit tests for the `handover-sim` package version."""

# SRL
import handover


def test_handover_sim_version() -> None:
    """Test `handover-sim` package version is set."""
    assert handover.__version__ is not None
    assert handover.__version__ != ""
