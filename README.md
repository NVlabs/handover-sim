# Handover-Sim

### Prerequisites

This code has so far been tested on Python 3.6.

### Installation

1. Clone the repo with `--recursive` and and cd into it:

    ```Shell
    git clone --recursive ssh://git@gitlab-master.nvidia.com:12051/ychao/handover-sim.git
    cd handover-sim
    ```

2. Install Python package and dependencies:

    ```Shell
    pip install -e .
    ```

3. Download data from OMG-Planner:

    ```Shell
    cd OMG-Planner
    ./download_data.sh
    cd ..
    ```

4. Compile YCB models:

    ```Shell
    python handover/data/compile_ycb_models.py
    ```

5. Download the DexYCB dataset and set up a symlink to the dataset folder:

    ```Shell
    cd handover/data
    ln -s $DEX_YCB_PATH dex-ycb
    cd ../..
    ```

    `$DEX_YCB_PATH` should be a folder with the following structure:

    ```Shell
    ├── 20200709-weiy/
    ├── 20200813-ceppner/
    ├── ...
    ├── calibration/
    └── models/
    ```

6. Cache DexYCB data:

    ```Shell
    python handover/data/cache_dex_ycb_data.py
    ```

### Running demos

```Shell
python examples/demo_handover_env.py
```
