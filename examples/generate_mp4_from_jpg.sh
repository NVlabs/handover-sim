#!/bin/bash

if [ -z $1 ]; then
  echo "Result dir is required."
  exit
else
  res_dir=$1
fi

for x in $(ls $res_dir | grep -E '^[0-9]{3}$'); do
  mp4_file=$res_dir/$x.mp4
  if [[ ! -f $mp4_file  ]]; then
    ffmpeg -y -framerate 60 -i $res_dir/$x/%*.jpg $mp4_file
  fi
done
