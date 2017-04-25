#!/usr/bin/env bash

# sudo apt install ppa-purge
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update -q
sudo apt-get -qq install g++-4.8 -y
sudo apt-get -qq install gcc-4.8 -y

# sudo apt-get update -qq
# sudo apt-get install -qq g++-4.8

# sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y 
# sudo apt-get -qq update -y
# sudo apt-get upgrade -y
# sudo apt-get dist-upgrade -y
sudo apt-get -qq install libstdc++6 -y
# sudo apt-get -f install

# if [ "$CXX" = "g++" ]; then export CXX="g++-4.9" CC="gcc-4.9"; fi

sudo apt-get install libgtest-dev
sudo apt-get install cmake
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp *.a /usr/lib 

git clone https://github.com/g-truc/glm.git
cd glm
git checkout 0.9.7 -b glm
mkdir build
cd build
cmake ..
make
make install
cd ../../
