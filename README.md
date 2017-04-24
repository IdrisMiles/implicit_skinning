# Implicit Skinning
This is my Advanced Graphics Software Development Techniques assignment for Bournemouth University.
This project is my implementation of implicit skinning based on ["Implicit Skinning: Real-Time Skin Deformation with Contact Modeling"](http://rodolphe-vaillant.fr/pivotx/templates/projects/implicit_skinning/implicit_skinning.pdf). 
The project is written in C++ and uses CUDA to speed up evaluation of field functions as well as perform skinning computations  

## Docs
To build the documentation using Doxygen, run:
```bash
doxygen Doxyfile
```

## Dependencies
| Name | Version |
| ---- | ------- |
|[CUDA](https://developer.nvidia.com/cuda-downloads) | >= 7 |
|[GLM](http://glm.g-truc.net/0.9.8/index.html)| >= 0.9.2 |
|[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)| >= 3.2.9 |


## Install and Build
```bash
git clone https://github.com/IdrisMiles/ImplicitSkinning.git
cd ImplicitSkinning
qmake
make clean
make
```

## Run
```bash
cd ImplicitSkinning/bin
./app
```

## Version
This is version 0.1.0

## Author

## License
