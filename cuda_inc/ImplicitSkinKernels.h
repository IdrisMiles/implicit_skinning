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
#include "ScalarField/composedfieldGPU.h"



/// @author Idris Miles
/// @version 1.0
/// @data 18/04/2017


/// @namespace isgw, Implicit Skin GPU Wrapper (isgw), wraps up Implicit Skinining CUDA kernals in a CPP interface
namespace isgw {


/// @brief Function to divide a by b and add 1 if there is a remainder,
/// useful for generatinig the number of CUDA blocks from size of data and number of threads
/// @param a : unsigned integer numerator
/// @param b : unsigned integer denominator
uint iDivUp(uint a, uint b);


/// @brief Function to launch CUDA Kernel to perform linear blend weight skinning
/// @param _deformedVert : Device pointer to the deformed vertices, this is the pointer that is mapped to the vertex buffer object
/// @param _origVert : Const device pointer to the original mesh vertices
/// @param _deformedNorms : Device pointer to the deformed normals, this is the pointer that is mapped to the normal buffer object
/// @param _origNorms : Const device pointer to the original mesh normals
/// @param _transforms : Const device pointer to the bone transforms
/// @param _boneId : const device pointer to the vertex bone IDs
/// @param _weights : const device pointer to the vertex bone weights
/// @param _numVerts : The number of vertices in the mesh
/// @param _numBones : The number of bones
void LinearBlendWeightSkin(glm::vec3 *_deformedVert,
                           const glm::vec3 *_origVert,
                           glm::vec3 *_deformedNorms,
                           const glm::vec3 *_origNorms,
                           const glm::mat4 *_transform,
                           const uint *_boneId,
                           const float *_weight,
                           const uint _numVerts,
                           const uint _numBones);


/// @brief Function to launch CUDA Kernel to evaluate the global field
/// @param _output : Device pointer to the output
/// @param _samplePoint : Const device pointer to the sample points
/// @param _numSamples: The number of samples
/// @param _textureSpace : Const device pointer to the texture space transform matrices, transform world space to texture space
/// @param _rigidTransform : Const device pointer to the rigid bone transform matrices
/// @param _fieldFuncs : Const device pointer to an array of textures holding the individual field functions
/// @param _numFields : The number of field function textures
void SimpleEvalGlobalField(float *_output,
                           const glm::vec3 *_samplePoint,
                           const uint _numSamples,
                           const glm::mat4 *_textureSpace,
                           const glm::mat4 *_rigidTransforms,
                           const cudaTextureObject_t *_fieldFuncs,
                           const uint _numFields);


/// @brief Function to launch CUDA Kernel to evaluate gradient of global field
void SimpleEvalGradGlobalField(float *_outputF,
                               glm::vec3 *_outputG,
                               const glm::vec3 *_samplePoint,
                               const uint _numSamples,
                               const glm::mat4 *_textureSpace,
                               const glm::mat4 *_rigidTransforms,
                               const cudaTextureObject_t *_fieldFuncs,
                               const cudaTextureObject_t *_fieldDerivs,
                               const uint _numFields);


/// @brief Function to launch CUDA Kernel to evaluate the global field
void EvalGlobalField(float *_output,
                      const glm::vec3 *_samplePoint,
                      const uint _numSamples,
                      const glm::mat4 *_textureSpace,
                      const glm::mat4 *_rigidTransforms,
                      const cudaTextureObject_t *_fieldFuncs,
                      const cudaTextureObject_t *_fieldDeriv,
                      const uint _numFields,
                      const cudaTextureObject_t *_compOps,
                     const cudaTextureObject_t *_compOpDerivs,
                      const cudaTextureObject_t *_theta,
                      const uint _numOps,
                      const ComposedFieldCuda *_compFields,
                      const uint _numCompFields);

/// @brief Function to launch CUDA Kernel to perform implicit skinning
void SimpleImplicitSkin(glm::vec3 *_deformedVert,
                          const glm::vec3 *_normal,
                          const float *_origIsoValue,
                        glm::vec3 *_prevIsoGrad,
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

/// @brief Method to generate scatter address
void GenerateScatterAddress(int *begin,
                            int *end,
                            int *scatteredAddr);

/// @brief Method to generate the one ring centroid weights for each mesh vertex
void GenerateOneRingCentroidWeights(glm::vec3 *d_verts, const glm::vec3 *d_normals,
                                    const uint _numVerts,
                                    float *_centroidWeights,
                                    const int *_oneRingIds,
                                    const glm::vec3 *_oneRingVerts,
                                    const int *_numNeighsPerVert,
                                    const int *_oneRingScatterAddr);

}
#endif //IMPLICITSKINKERNELS_H
