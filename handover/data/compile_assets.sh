#!/bin/bash

DIR="$( dirname "${BASH_SOURCE[0]}" )"
cd $DIR

python compile_table_assets.py

python compile_ycb_assets.py
