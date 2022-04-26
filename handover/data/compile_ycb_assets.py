import os

ycb_classes = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick",
]

urdf_str = """<?xml version="1.0"?>
<robot name="model_normalized">
  <link name="link0">
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <link name="link1">
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="joint1" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="link0"/>
    <child link="link1"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1000" upper="1000"/>
  </joint>
  <link name="link2">
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="joint2" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="link1"/>
    <child link="link2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1000" upper="1000"/>
  </joint>
  <link name="link3">
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="joint3" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="link2"/>
    <child link="link3"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1000" upper="1000"/>
  </joint>
  <link name="link4">
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="joint4" type="continuous">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="link3"/>
    <child link="link4"/>
    <axis xyz="1 0 0"/>
  </joint>
  <link name="link5">
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="joint5" type="continuous">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="link4"/>
    <child link="link5"/>
    <axis xyz="0 1 0"/>
  </joint>
  <link name="link6">
    <contact>
      <lateral_friction value="0.9"/>
    </contact>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="textured_simple.obj" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="model_normalized_convex.obj" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
  <joint name="joint6" type="continuous">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="link5"/>
    <child link="link6"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""

asset_root = os.path.join(os.path.dirname(__file__), "assets")


def main():
    print("Compiling YCB assets")

    for x in ycb_classes:
        print("{}".format(x))
        urdf_dir = os.path.join(asset_root, x)
        os.makedirs(urdf_dir, exist_ok=True)

        urdf_file = os.path.join(urdf_dir, "model_normalized.urdf")
        if not os.path.isfile(urdf_file):
            with open(urdf_file, "w") as f:
                f.write(urdf_str)
        else:
            with open(urdf_file, "r") as f:
                assert f.read() == urdf_str

    print("Done.")


if __name__ == "__main__":
    main()
