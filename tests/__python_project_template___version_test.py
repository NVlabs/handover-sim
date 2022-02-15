# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Unit tests for the `__python_project_template__` package version."""

# SRL
import __python_project_template__


def test___python_project_template___version() -> None:
    """Test `__python_project_template__` package version is set."""
    assert __python_project_template__.__version__ is not None
    assert __python_project_template__.__version__ != ""
