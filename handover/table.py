# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import easysim
import os


class Table:
    def __init__(self, cfg, scene):
        self._cfg = cfg
        self._scene = scene

        body = easysim.Body()
        body.name = "table"
        body.geometry_type = easysim.GeometryType.URDF
        body.urdf_file = os.path.join(
            os.path.dirname(__file__), "data", "assets", "table", "table.urdf"
        )
        body.use_fixed_base = True
        body.initial_base_position = (
            self._cfg.ENV.TABLE_BASE_POSITION + self._cfg.ENV.TABLE_BASE_ORIENTATION
        )
        body.link_color = [(1.0, 1.0, 1.0, 1.0)]
        body.link_collision_filter = [self._cfg.ENV.COLLISION_FILTER_TABLE]
        self._scene.add_body(body)
        self._body = body

    @property
    def body(self):
        return self._body
