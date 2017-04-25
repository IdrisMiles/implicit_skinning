#!/usr/bin/env bash

git clone https://github.com/g-truc/glm.git
cd glm
git checkout 0.9.8.0 -b glm
mkdir build
cd build
cmake ..
make
make install
cd ../../

sudo apt-get update
sudo apt-get install -y qt5-qmake