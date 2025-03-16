#!/bin/bash

set -ex

CYDIR=$(git rev-parse --show-toplevel)
G_DIR=$CYDIR/generators/gemmini/software/gemmini-rocc-tests
O_DIR=$CYDIR/software/tutorial/overlay/root

echo "Building Gemmini RoCC tests"
cd $G_DIR

./build.sh transformers
cd build
rm -rf $O_DIR
mkdir -p $O_DIR
cp -r transformers $O_DIR/

echo "Complete!"
