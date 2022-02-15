# Handover-Sim

## Prerequisites

This code is tested with Python 3.8 on Linux.

## Installation

For good practice for Python package management, it is recommended to install the package into a virtual environment (e.g., `virtualenv` or `conda`).

1. Clone the repo with `--recursive` and and cd into it:

    ```Shell
    git clone --recursive ssh://git@gitlab-master.nvidia.com:12051/ychao/handover-sim.git
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

4. Download data from OMG-Planner:

    ```Shell
    cd OMG-Planner
    ./download_data.sh
    cd ..
    ```

5. Download MANO models and code (`mano_v1_2.zip`) from the [MANO website](https://mano.is.tue.mpg.de) and place the file under `handover/data`. Unzip the file:

    ```Shell
    cd handover/data
    unzip mano_v1_2.zip
    cd ../..
    ```

6. Download the DexYCB dataset.

    **Option 1**: Download cached dataset: **(recommended)**

    ```Shell
    cd handover/data
    # Download dex-ycb-cache_20210423.tar.gz from https://drive.google.com/file/d/1M2SZ5kLZuKy4aUrDuXllvTOyXy8OKLq4.
    tar -zxvf dex-ycb-cache_20210423.tar.gz
    cd ../..
    ```

    **Option 2**: Download full dataset and cache the data:

    1.  Download the DexYCB dataset from the [DexYCB project site](https://dex-ycb.github.io).

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

7. Compile assets:

    ```Shell
    ./handover/data/compile_assets.sh
    ```

## Running Examples

1. Running a handover environment:

    ```Shell
    python examples/demo_handover_env.py
    ```

2. Running a planned trajectory:

    ```Shell
    python examples/demo_trajectory.py
    ```

3. Running a benchmark environment:

    ```Shell
    python examples/demo_benchmark_env.py
    ```
