# Handover-Sim

## Prerequisites

This code is tested with Python 3.8 on Ubuntu 20.04.

## Installation

For good practice for Python package management, it is recommended to install the package into a virtual environment (e.g., `virtualenv` or `conda`).

1. Clone the repo with `--recursive` and and cd into it:

    ```Shell
    git clone --recursive git@github.com:NVlabs/handover-sim.git
    cd handover-sim
    ```

2. Install `easysim`:

    ```Shell
    git clone ssh://git@gitlab-master.nvidia.com:12051/ychao/easysim.git
    pip install -e ./easysim
    ```

3. Install `handover-sim` as a Python package:

    ```Shell
    pip install -e .
    ```

4. Download MANO models and code (`mano_v1_2.zip`) from the [MANO website](https://mano.is.tue.mpg.de) and place the file under `handover/data/`. Unzip with:

    ```Shell
    cd handover/data
    unzip mano_v1_2.zip
    cd ../..
    ```

    This will extract a folder `handover/data/mano_v1_2/`.

5. Download the DexYCB dataset.

    **Option 1**: Download cached dataset: **(recommended)**

    1. Download [`dex-ycb-cache-20220323.tar.gz`](https://drive.google.com/uc?export=download&id=1Jqe2iqI7inoEdE3BL4vEs25eT5M7aUHd) (507M) and place the file under `handover/data/`. Extract with:

        ```Shell
        cd handover/data
        tar zxvf dex-ycb-cache-20220323.tar.gz
        cd ../..
        ```

        This will extract a folder `handover/data/dex-ycb-cache/`.

    **Option 2**: Download full dataset and cache the data:

    1. Download the DexYCB dataset from the [DexYCB project site](https://dex-ycb.github.io).

    2. Set the environment variable for dataset path:

        ```Shell
        export DEX_YCB_DIR=/path/to/dex-ycb
        ```

        `$DEX_YCB_DIR` should be a folder with the following structure:

        ```Shell
        ├── 20200709-subject-01/
        ├── 20200813-subject-02/
        ├── ...
        ├── calibration/
        └── models/
        ```

    3. Cache the dataset:

        ```Shell
        python handover/data/cache_dex_ycb_data.py
        ```

        The cached dataset will be saved to `handover/data/dex-ycb-cache/`.

6. Compile assets.

    1. Download [`assets-3rd-party-20220511.tar.gz`](https://drive.google.com/uc?export=download&id=1tDiXvW5vwJDOCgK61VEsFaZ7Z00gF0vj) (155M) and place the file under `handover/data/`. Extract with:

        ```Shell
        cd handover/data
        tar zxvf assets-3rd-party-20220511.tar.gz
        cd ../..
        ```

        This will extract a folder `handover/data/assets/` with 3rd party assets. See [handover/data/README.md](./handover/data/README.md) for the source of these assets.

    2. Compile assets:

        ```Shell
        ./handover/data/compile_assets.sh
        ```

        The compiled assets will be saved to `handover/data/assets/`.

## Running Examples

1. Running a handover environment:

    ```Shell
    python examples/demo_handover_env.py \
      SIM.RENDER True
    ```

2. Running a planned trajectory:

    ```Shell
    python examples/demo_trajectory.py \
      SIM.RENDER True
    ```

3. Running a benchmark wrapper:

    ```Shell
    python examples/demo_benchmark_wrapper.py \
      SIM.RENDER True
    ```
