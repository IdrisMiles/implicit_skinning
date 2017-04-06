#ifndef IMPLICITSKINKERNELS_H
#define IMPLICITSKINKERNELS_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtx/vector_angle.hpp>

#include "cutil_math.h"
#include "ScalarField/compfield.h"

#include <stdio.h>


/// @author Idris Miles
/// @version 1.0

namespace kernels {

/// @brief Function
uint iDivUp(uint a, uint b);


/// @brief Function to launch CUDA Kernel to perform linear blend weight skinning
void LinearBlendWeightSkin(glm::vec3 *_deformedVert,
                           const glm::vec3 *_origVert,
                           const glm::mat4 *_transform,
                           const uint *_boneId,
                           const float *_weight,
                           const uint _numVerts,
                           const uint _numBones);

void SimpleEvalGlobalField(float *_output,
                            const glm::vec3 *_samplePoint,
                            const uint _numSamples,
                            const glm::mat4 *_textureSpace,
                            const glm::mat4 *_rigidTransforms,
                            const cudaTextureObject_t *_fieldFuncs,
                            const uint _numFields);

void EvalGlobalField(float *_output,
                      const glm::vec3 *_samplePoint,
                      const uint _numSamples,
                      const glm::mat4 *_textureSpace,
                      const glm::mat4 *_rigidTransforms,
                      const cudaTextureObject_t *_fieldFuncs,
                      const cudaTextureObject_t *_fieldDeriv,
                      const uint _numFields,
                      const cudaTextureObject_t *_compOps,
                      const cudaTextureObject_t *_theta,
                      const uint _numOps,
                      const ComposedFieldCuda *_compFields,
                      const uint _numCompFields);

void SimpleImplicitSkin(glm::vec3 *_deformedVert,
                          const glm::vec3 *_normal,
                          const float *_origIsoValue,
                          const uint _numVerts,
                          const glm::mat4 *_textureSpace,
                          const glm::mat4 *_rigidTransforms,
                          const cudaTextureObject_t *_fieldFuncs,
                          const cudaTextureObject_t *_fieldDeriv,
                          const uint _numFields,
                          const int *_oneRingNeigh,
                          const float *_centroidWeights,
                          const int *_numNeighs,
                          const int *_neighScatterAddr);

void GenerateScatterAddress(int *begin,
                            int *end,
                            int *scatteredAddr);

void GenerateOneRingCentroidWeights(glm::vec3 *d_verts,
                                    const uint _numVerts,
                                    float *_centroidWeights,
                                    const int *_oneRingIds,
                                    const int *_numNeighsPerVert,
                                    const int *_oneRingScatterAddr);

}
#endif //IMPLICITSKINKERNELS_H
