#!/usr/bin/env bash


sudo apt-get install libgtest-dev
sudo apt-get install cmake
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp *.a /usr/lib 

git clone https://github.com/g-truc/glm.git
cd glm
git checkout 0.9.8 -b glm
mkdir build
cd build
cmake ..
make
make install
cd ../../
