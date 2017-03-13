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

/// @brief CUDA Kernel to evalute global field function containing implicit skin
__global__ void EvaluateGlobalField(float *_output,
                                    glm::vec3 *_samplePoint,
                                    uint _numSamples,
                                    glm::mat4 _textureSpace,
                                    glm::mat4 *_rigidTransforms,
                                    cudaTextureObject_t *_fieldFuncs,
                                    cudaTextureObject_t *_fieldDeriv,
                                    uint _numFields,
                                    cudaTextureObject_t *_compOps,
                                    cudaTextureObject_t *_theta, // opening function
                                    uint _numOps,
                                    CompField *_compFields,
                                    uint _numCompFields);


/// @brief CUDA Kernel to perform linear blend weight skinning
__global__ void LinearBlendWeightSkin(glm::vec3 *_deformedVert,
                                      const glm::vec3 *_origVert,
                                      const glm::mat4 *_transform,
                                      const uint *_boneId,
                                      const float *_weight,
                                      const uint _numVerts,
                                      const uint _numBones);

#endif //IMPLICITSKINKERNELS_H
