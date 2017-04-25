#!/usr/bin/env bash

# sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
# sudo apt-get update -q

# sudo apt-get install g++-4.8 -y

sudo apt-get install libstdc++6
sudo add-apt-repository ppa:ubuntu-toolchain-r/test 
sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade

sudo apt-get install libgtest-dev
sudo apt-get install cmake
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp *.a /usr/lib 

git clone https://github.com/g-truc/glm.git
cd glm
git checkout 0.9.8.0 -b glm
mkdir build
cd build
cmake ..
make
make install
cd ../../
