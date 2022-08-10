#!/bin/bash

DIR="$( dirname "${BASH_SOURCE[0]}" )"
cd $DIR

FILE=icra2022_results.tar.gz
ID=1OnU9HMutlfBgv9wE2HOIbavZLKFSZr__
CHECKSUM=8876c3853db101998510e4804e0bd96a

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading icra 2022 results (18M)..."

wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=$ID" -O $FILE

echo "Unzipping..."

tar zxvf $FILE

echo "Done."
