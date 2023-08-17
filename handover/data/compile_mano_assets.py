# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import os
import xml.etree.ElementTree as ET
import numpy as np

from mano_pybullet.hand_model import HandModel45
from mano_pybullet.mesh_utils import filter_mesh, save_mesh_obj
from scipy.spatial.transform import Rotation as Rot
from xml.dom import minidom

from handover.config import get_config_from_args
from handover.dex_ycb import DexYCB

mano_sides = ("right", "left")

models_dir = os.path.join(os.path.dirname(__file__), "mano_v1_2", "models")

asset_root = os.path.join(os.path.dirname(__file__), "assets")


def is_integer_num(x):
    if isinstance(x, int):
        return True
    if isinstance(x, float):
        return x.is_integer()
    return False


def num_list_to_string(x):
    try:
        iter(x)
    except TypeError:
        x = [x]

    if all([is_integer_num(y) for y in x]):
        return " ".join(["{:.0f}"] * len(x)).format(*x)
    else:
        return " ".join(["{:.64f}"] * len(x)).format(*x)


def create_link(
    name,
    mass,
    inertial_xyz=[0, 0, 0],
    inertial_rpy=[0, 0, 0],
    visual_filename=None,
    visual_xyz=[0, 0, 0],
    visual_rpy=[0, 0, 0],
    collision_filename=None,
    collision_xyz=[0, 0, 0],
    collision_rpy=[0, 0, 0],
):
    link = ET.Element("link", name=name)

    inertial = ET.SubElement(link, "inertial")

    ET.SubElement(
        inertial,
        "origin",
        xyz=num_list_to_string(inertial_xyz),
        rpy=num_list_to_string(inertial_rpy),
    )
    ET.SubElement(inertial, "mass", value=num_list_to_string(mass))
    ET.SubElement(
        inertial,
        "inertia",
        ixx="0" if mass == 0 else "1",
        ixy="0",
        ixz="0",
        iyy="0" if mass == 0 else "1",
        iyz="0",
        izz="0" if mass == 0 else "1",
    )

    if visual_filename is not None:
        visual = ET.SubElement(link, "visual")

        ET.SubElement(
            visual, "origin", xyz=num_list_to_string(visual_xyz), rpy=num_list_to_string(visual_rpy)
        )
        geometry = ET.SubElement(visual, "geometry")

        ET.SubElement(geometry, "mesh", filename=visual_filename)

    if collision_filename is not None:
        collision = ET.SubElement(link, "collision")

        ET.SubElement(
            collision,
            "origin",
            xyz=num_list_to_string(collision_xyz),
            rpy=num_list_to_string(collision_rpy),
        )
        geometry = ET.SubElement(collision, "geometry")

        ET.SubElement(geometry, "mesh", filename=collision_filename)

    return link


def create_joint(name, type_, xyz, rpy, parent, child, axis, limit):
    joint = ET.Element("joint", name=name, type=type_)

    ET.SubElement(joint, "origin", xyz=num_list_to_string(xyz), rpy=num_list_to_string(rpy))
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "axis", xyz=num_list_to_string(axis))
    ET.SubElement(
        joint, "limit", lower=num_list_to_string(limit[0]), upper=num_list_to_string(limit[1])
    )

    return joint


def make_link_mesh(collision, link_index, model):
    threshold = 0.2
    if collision and link_index in [4, 7, 10]:
        threshold = 0.7
    vertex_mask = model.weights[:, link_index] > threshold
    vertices = model.vertices(model._betas)
    vertices, faces = filter_mesh(vertices, model.faces, vertex_mask)
    vertices -= model.joints[link_index].origin

    return vertices, faces


def make_link_mesh_filename(collision, mesh_name):
    if collision:
        return os.path.join("meshes", "collision", "{}.obj".format(mesh_name))
    else:
        return os.path.join("meshes", "visual", "{}.obj".format(mesh_name))


def main():
    print("Compiling MANO assets")

    cfg = get_config_from_args()

    dex_ycb = DexYCB(cfg)

    subjects = []

    for scene_id in range(dex_ycb.NUM_SEQUENCES):
        scene_data = dex_ycb.get_scene_data(scene_id)

        n = scene_data["name"].split("/")[0]
        if n in subjects:
            continue

        mano_betas = scene_data["mano_betas"][0]

        for mano_side in mano_sides:
            model = HandModel45(
                left_hand=mano_side == "left", models_dir=models_dir, betas=mano_betas
            )

            print("{}: {}".format(n, mano_side))
            urdf_dir = os.path.join(asset_root, "{}_{}".format(n, mano_side))
            os.makedirs(urdf_dir, exist_ok=True)

            robot = ET.Element("robot", name="mano")

            robot.append(create_link("link0", 0))

            robot.append(create_link("link1", 0))
            robot.append(
                create_joint(
                    "joint1",
                    "prismatic",
                    [0, 0, 0],
                    [0, 0, 0],
                    "link0",
                    "link1",
                    [1, 0, 0],
                    [-1000, +1000],
                )
            )
            robot.append(create_link("link2", 0))
            robot.append(
                create_joint(
                    "joint2",
                    "prismatic",
                    [0, 0, 0],
                    [0, 0, 0],
                    "link1",
                    "link2",
                    [0, 1, 0],
                    [-1000, +1000],
                )
            )
            robot.append(create_link("link3", 0))
            robot.append(
                create_joint(
                    "joint3",
                    "prismatic",
                    [0, 0, 0],
                    [0, 0, 0],
                    "link2",
                    "link3",
                    [0, 0, 1],
                    [-1000, +1000],
                )
            )
            robot.append(create_link("link4", 0))
            robot.append(
                create_joint(
                    "joint4",
                    "continuous",
                    [0, 0, 0],
                    [0, 0, 0],
                    "link3",
                    "link4",
                    [1, 0, 0],
                    [+1, -1],
                )
            )
            robot.append(create_link("link5", 0))
            robot.append(
                create_joint(
                    "joint5",
                    "continuous",
                    [0, 0, 0],
                    [0, 0, 0],
                    "link4",
                    "link5",
                    [0, 1, 0],
                    [+1, -1],
                )
            )
            robot.append(create_link("link6", 0))
            robot.append(
                create_joint(
                    "joint6",
                    "continuous",
                    [0, 0, 0],
                    [0, 0, 0],
                    "link5",
                    "link6",
                    [0, 0, 1],
                    [+1, -1],
                )
            )

            link_counter = 6
            mesh_counter = 0

            meshes = []

            joints = model.joints

            link_name = "link{:d}".format(link_counter + 1)
            joint_name = "joint{:d}".format(link_counter + 1)
            parent = "link{:d}".format(link_counter)
            mass = 0.2
            visual_filename = make_link_mesh_filename(False, "mesh{:d}".format(mesh_counter + 1))
            collision_filename = make_link_mesh_filename(True, "mesh{:d}".format(mesh_counter + 1))
            meshes.append(
                (make_link_mesh(False, 0, model), os.path.join(urdf_dir, visual_filename))
            )
            meshes.append(
                (make_link_mesh(True, 0, model), os.path.join(urdf_dir, collision_filename))
            )
            mesh_counter += 1
            shape_rpy = Rot.from_matrix(joints[0].basis.T).as_euler("xyz")
            xyz = joints[0].origin
            mat = joints[0].basis
            rpy = Rot.from_matrix(mat).as_euler("xyz")
            axis = [0, 0, 0]
            robot.append(
                create_link(
                    link_name,
                    mass,
                    inertial_xyz=[0, 0, 0],
                    inertial_rpy=[0, 0, 0],
                    visual_filename=visual_filename,
                    visual_xyz=[0, 0, 0],
                    visual_rpy=shape_rpy,
                    collision_filename=collision_filename,
                    collision_xyz=[0, 0, 0],
                    collision_rpy=shape_rpy,
                )
            )
            robot.append(
                create_joint(joint_name, "fixed", xyz, rpy, parent, link_name, axis, [+1, -1])
            )
            link_mapping = {0: link_counter}
            link_counter += 1

            for i, j in model.kintree_table.T[1:]:
                parent_index = link_mapping[i]
                for k, (axis, limits) in enumerate(zip(joints[j].axes, joints[j].limits)):
                    link_name = "link{:d}".format(link_counter + 1)
                    joint_name = "joint{:d}".format(link_counter + 1)
                    parent = "link{:d}".format(parent_index + 1)
                    if k != len(joints[j].axes) - 1:
                        mass = 0
                        visual_filename = None
                        collision_filename = None
                        shape_rpy = [0, 0, 0]
                    else:
                        mass = 0.02
                        visual_filename = make_link_mesh_filename(
                            False, "mesh{:d}".format(mesh_counter + 1)
                        )
                        collision_filename = make_link_mesh_filename(
                            True, "mesh{:d}".format(mesh_counter + 1)
                        )
                        meshes.append(
                            (
                                make_link_mesh(False, j, model),
                                os.path.join(urdf_dir, visual_filename),
                            )
                        )
                        meshes.append(
                            (
                                make_link_mesh(True, j, model),
                                os.path.join(urdf_dir, collision_filename),
                            )
                        )
                        mesh_counter += 1
                        shape_rpy = Rot.from_matrix(joints[j].basis.T).as_euler("xyz")
                    if k == 0:
                        xyz = joints[i].basis.T @ (joints[j].origin - joints[i].origin)
                        mat = joints[i].basis.T @ joints[j].basis
                    else:
                        xyz = [0, 0, 0]
                        mat = np.eye(3)
                    rpy = Rot.from_matrix(mat).as_euler("xyz")
                    axis = np.eye(3)[ord(axis) - ord("x")]
                    robot.append(
                        create_link(
                            link_name,
                            mass,
                            inertial_xyz=[0, 0, 0],
                            inertial_rpy=[0, 0, 0],
                            visual_filename=visual_filename,
                            visual_xyz=[0, 0, 0],
                            visual_rpy=shape_rpy,
                            collision_filename=collision_filename,
                            collision_xyz=[0, 0, 0],
                            collision_rpy=shape_rpy,
                        )
                    )
                    robot.append(
                        create_joint(
                            joint_name, "revolute", xyz, rpy, parent, link_name, axis, limits
                        )
                    )
                    parent_index = link_counter
                    link_counter += 1
                link_mapping[j] = parent_index

            urdf_str = minidom.parseString(ET.tostring(robot)).toprettyxml(indent="  ")

            # Save urdf file.
            urdf_file = os.path.join(urdf_dir, "mano.urdf")
            if not os.path.isfile(urdf_file):
                with open(urdf_file, "w") as f:
                    f.write(urdf_str)
            else:
                with open(urdf_file, "r") as f:
                    assert f.read() == urdf_str

            # Save mesh files.
            for (vertices, faces), obj_file in meshes:
                obj_dir = os.path.dirname(obj_file)
                os.makedirs(obj_dir, exist_ok=True)

                if not os.path.isfile(obj_file):
                    save_mesh_obj(obj_file, vertices, faces)
                else:
                    obj_lines = []
                    for vert in vertices:
                        obj_lines.append("v {:f} {:f} {:f}".format(*vert))
                    for face in faces + 1:
                        obj_lines.append("f {:d} {:d} {:d}".format(*face))
                    with open(obj_file, "r") as f:
                        lines = [line.rstrip("\n") for line in f]
                    assert lines == obj_lines

        subjects.append(n)

    print("Done.")


if __name__ == "__main__":
    main()
