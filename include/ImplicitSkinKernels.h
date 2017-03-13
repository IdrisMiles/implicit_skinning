#ifndef IMPLICITSKINKERNELS_H
#define IMPLICITSKINKERNELS_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtx/vector_angle.hpp>

#include "cutil_math.h"
#include "ScalarField/compfield.h"

#include <stdio.h>


/// @author Idris Miles
/// @version 1.0


/// @brief Function
uint iDivUp(uint a, uint b);

void InitCUDAMemory();


/// @brief Function to launch CUDA Kernel to perform linear blend weight skinning
void LinearBlendWeightSkin(glm::vec3 *_deformedVert,
                           const glm::vec3 *_origVert,
                           const glm::mat4 *_transform,
                           const uint *_boneId,
                           const float *_weight,
                           const uint _numVerts,
                           const uint _numBones);

#endif //IMPLICITSKINKERNELS_H
