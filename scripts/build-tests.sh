#!/usr/bin/env bash

locate / | grep libgtest.

cd test/Mesh
# qmake
make clean
make
cd ../bin
./TestMesh