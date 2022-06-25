#!/bin/bash

DIR="$( dirname "${BASH_SOURCE[0]}" )"
cd $DIR

FILE=icra2022_results.zip
ID=10I7EN_3yWOly_k_LnZg0nOxfDOxgnmCi
CHECKSUM=e57c15eb9aa71c1e1fd14e03c66e63d8

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
