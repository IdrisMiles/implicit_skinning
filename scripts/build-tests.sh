#!/usr/bin/env bash

cd test/Mesh
# qmake
make clean
make
cd ../bin
./TestMesh