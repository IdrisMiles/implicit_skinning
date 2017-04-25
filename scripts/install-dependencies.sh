#!/usr/bin/env bash

sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update -q
sudo apt-get install g++-5 -y 

git clone https://github.com/g-truc/glm.git
cd glm
git checkout 0.9.8.0 -b glm
mkdir build
cd build
cmake ..
make
make install
cd ../../
