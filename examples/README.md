### Contents

1. [Installation: OMG-Planner](#installation-omg-planner)
2. [Installation: GA-DDPG](#installation-ga-ddpg)

## Installation: OMG-Planner

Below is our installation script for OMG-Planner.
- This script is tested with **Python 3.8** on **Ubuntu 20.04**.
- See [OMG-Planner](https://github.com/liruiw/OMG-Planner) for more information.

```Shell
# Install Ubuntu packages.
# - libassimp-dev is required for pyassimp.
# - libegl-dev is required for ycb_renderer.
# - libgles2 is required for ycb_renderer.
# - libglib2.0-0 is required for opencv-python.
# - libxslt1-dev is required for lxml.
apt install \
    libassimp-dev \
    libegl-dev \
    libgles2 \
    libglib2.0-0 \
    libxslt1-dev

# The script below should be ran under handover-sim/.

# Clone OMG-Planner.
# - All our experiements were ran on commit a3b8b683273327c63092a1454d58279e0e0be9de.
git clone --recursive https://github.com/liruiw/OMG-Planner.git
cd OMG-Planner
git checkout a3b8b68

# Install Python packages in requirements.txt.
sed -i "s/opencv-python==3.4.3.18/opencv-python/g" requirements.txt
sed -i "s/torch==1.4.0/torch/g" requirements.txt
sed -i "s/torchvision==0.4.2/torchvision/g" requirements.txt
pip install -r requirements.txt

# Install ycb_render.
cd ycb_render
python setup.py develop
cd ..

# Install eigen.
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
git checkout 3.4.0
mkdir -p release && mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$( cd ../release && pwd )
make -j8
make install
cd ../..

# Install Sophus.
cd Sophus
mkdir -p release && mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$( cd ../release && pwd ) \
  -DEIGEN3_INCLUDE_DIR=$( cd ../../eigen/release/include/eigen3 && pwd )
make -j8
make install
cd ../..

# Install layers.
cd layers
sed -i "s@/usr/local/include/eigen3\", \"/usr/local/include@$( cd ../eigen/release/include/eigen3 && pwd )\", \"$( cd ../Sophus/release/include && pwd )@g" setup.py
python setup.py install
cd ..

# Install PyKDL.
cd orocos_kinematics_dynamics
cd sip-4.19.3
python configure.py
make -j8
make install
cd ../orocos_kdl
mkdir -p release && mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$( cd ../release && pwd ) \
  -DEIGEN3_INCLUDE_DIR=$( cd ../../../eigen/release/include/eigen3 && pwd )
make -j8
make install
cd ../../python_orocos_kdl
mkdir -p build && cd build
# ** YOU NEED TO MODIFY $VIRTUAL_ENV BELOW. **
cmake .. \
  -DPYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python \
  -DCMAKE_PREFIX_PATH=$( cd ../../orocos_kdl/release && pwd )
make -j8
cp PyKDL.so $VIRTUAL_ENV/lib/python3.8/site-packages
cd ../../..

# Download data.
./download_data.sh

cd ..
```

## Installation: GA-DDPG

Below is our installation script for GA-DDPG.
- This script is tested with **Python 3.8** on **Ubuntu 20.04**.
- See [GA-DDPG](https://github.com/liruiw/GA-DDPG) for more information.

```Shell
# The script below should be ran under handover-sim/.

# Clone OMG-Planner.
# - All our experiements were ran on commit 6d2b10ffb81c125536740c82df23283d8a1c3ac8.
git clone --recursive https://github.com/liruiw/GA-DDPG.git
cd GA-DDPG
git checkout 6d2b10f

# Install Python packages in requirements.txt.
sed -i "s/opencv-python==3.4.3.18/opencv-python/g" requirements.txt
sed -i "s/tabulate==0.8.6/tabulate/g" requirements.txt
sed -i "s/torch==1.4.0/torch/g" requirements.txt
sed -i "s/torchvision==0.5.0/torchvision/g" requirements.txt
pip install -r requirements.txt

# Clone Pointnet2_PyTorch.
# - All our experiements were ran on commit e803915c929b3b69bafe4c07e1f2c322e7a20aae.
git clone https://github.com/liruiw/Pointnet2_PyTorch
cd Pointnet2_PyTorch
git checkout e803915

# Install Python packages in requirements.txt.
pip install -r requirements.txt
cd ..

# Download model.
bash experiments/scripts/download_model.sh

cd ..
```
