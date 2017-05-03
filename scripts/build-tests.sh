#!/usr/bin/env bash

cd test/Mesh
# qmake
make clean
make

cd ../Texture3DCpu
make clean
make

cd ../bin
./TestMesh
./TestTexture3DCpu