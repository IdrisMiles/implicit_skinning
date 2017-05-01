#ifndef IMPLICITSKINKERNELS_H
#define IMPLICITSKINKERNELS_H

//---------------------------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------------------------

/// @author Idris Miles
/// @version 1.0
/// @data 18/04/2017

//---------------------------------------------------------------------------------------------------------------------------------

__device__ void VertexProjection(glm::vec3 &_deformedVert,
                                 const float &_origIso,
                                 const float &_newIso,
                                 const glm::vec3 &_newIsoGrad,
                                 glm::vec3 &_prevIsoGrad,
                                 float &_gradAngle,
                                 const float &_sigma,
                                 const float &_contactAngle);


__device__ glm::vec3 ProjectPointOnToPlane(const glm::vec3 &_point,
                                           const glm::vec3 &_planeOrigin,
                                           const glm::vec3 &_planeNormal);


__device__ void TangentialRelaxation (glm::vec3 &_deformedVert,
                                     const glm::vec3 &_normal,
                                     const float _origIso,
                                     const float _newIso,
                                      glm::vec3 *_verts,
                                     const int *_oneRingNeigh,
                                     const float *_centroidWeights,
                                     const int _numNeighs);


__device__ void LaplacianSmoothing(glm::vec3 &_deformedVert,
                                   const glm::vec3 &_normal,
                                   const int *_oneRingNeigh,
                                   const float *_centroidWeights,
                                   const int _numNeighs,
                                   const glm::vec3 *_verts,
                                   const float _beta);


__device__ void EvalGlobalField(float &_outputF,
                                const glm::vec3 &_samplePoint,
                                const int _numSamples,
                                const glm::mat4 *_textureSpace,
                                const glm::mat4 *_rigidTransforms,
                                const cudaTextureObject_t *_fieldFuncs,
                                const int _numFields,
                                const cudaTextureObject_t *_compOps,
                                const cudaTextureObject_t *_theta,
                                const int _numOps,
                                const ComposedFieldCuda *_compFields,
                                const int _numCompFields);


__device__ void EvalGradGlobalField(float &_outputF,
                                glm::vec3 &_outputG,
                                const glm::vec3 &_samplePoint,
                                const int _numSamples,
                                const glm::mat4 *_textureSpace,
                                const glm::mat4 *_rigidTransforms,
                                const cudaTextureObject_t *_fieldFuncs,
                                const int _numFields,
                                const cudaTextureObject_t *_compOps,
                                const cudaTextureObject_t *_theta,
                                const int _numOps,
                                const ComposedFieldCuda *_compFields,
                                const int _numCompFields);

//---------------------------------------------------------------------------------------------------------------------------------

__global__ void EvalGlobalField_Kernel(float *_output,
                                       const glm::vec3 *_samplePoint,
                                       const int _numSamples,
                                       const glm::mat4 *_textureSpace,
                                       const glm::mat4 *_rigidTransforms,
                                       const cudaTextureObject_t *_fieldFuncs,
                                       const int _numFields,
                                       const cudaTextureObject_t *_compOps,
                                       const cudaTextureObject_t *_theta,
                                       const int _numOps,
                                       const ComposedFieldCuda *_compFields,
                                       const int _numCompFields);



__global__ void EvalGradGlobalField_Kernel(float *_output,
                                           glm::vec3 *_outputG,
                                       const glm::vec3 *_samplePoint,
                                       const int _numSamples,
                                       const glm::mat4 *_textureSpace,
                                       const glm::mat4 *_rigidTransforms,
                                       const cudaTextureObject_t *_fieldFuncs,
                                       const int _numFields,
                                       const cudaTextureObject_t *_compOps,
                                       const cudaTextureObject_t *_theta,
                                       const int _numOps,
                                       const ComposedFieldCuda *_compFields,
                                       const int _numCompFields);



__global__ void LinearBlendWeightSkin_Kernel(glm::vec3 *_deformedVert,
                                             const glm::vec3 *_origVert,
                                             glm::vec3 *_deformedNorms,
                                             const glm::vec3 *_origNorms,
                                             const glm::mat4 *_transform,
                                             const uint *_boneId,
                                             const float *_weight,
                                             const int _numVerts,
                                             const int _numBones);


__global__ void VertexProjection_Kernel(glm::vec3 *_deformedVert,
                                        const glm::vec3 *_normal,
                                        const float *_origIsoValue,
                                        glm::vec3 *_prevIsoGrad,
                                        const int _numVerts,
                                        const glm::mat4 *_textureSpace,
                                        const glm::mat4 *_rigidTransforms,
                                        const cudaTextureObject_t *_fieldFuncs,
                                        const int _numFields,
                                        const cudaTextureObject_t *_compOps,
                                        const cudaTextureObject_t *_theta,
                                        const int _numOps,
                                        const ComposedFieldCuda *_compFields,
                                        const int _numCompFields,
                                        const float _sigma,
                                        const float _contactAngle);


__global__ void TangentialRelaxation_Kernel(glm::vec3 *_deformedVert,
                                            const glm::vec3 *_normal,
                                            const float *_origIsoValue,
                                            glm::vec3 *_prevIsoGrad,
                                            const int _numVerts,
                                            const glm::mat4 *_textureSpace,
                                            const glm::mat4 *_rigidTransforms,
                                            const cudaTextureObject_t *_fieldFuncs,
                                            const int _numFields,
                                            const cudaTextureObject_t *_compOps,
                                            const cudaTextureObject_t *_theta,
                                            const int _numOps,
                                            const ComposedFieldCuda *_compFields,
                                            const int _numCompFields,
                                            const int *_oneRingVerts,
                                            const float *_centroidWeights,
                                            const int *_oneRingScatterAddr,
                                            const float _sigma,
                                            const float _contactAngle);


__global__ void LaplacianRelaxation_Kernel(glm::vec3 *_deformedVert,
                                            const glm::vec3 *_normal,
                                            const float *_origIsoValue,
                                            glm::vec3 *_prevIsoGrad,
                                            const int _numVerts,
                                            const glm::mat4 *_textureSpace,
                                            const glm::mat4 *_rigidTransforms,
                                            const cudaTextureObject_t *_fieldFuncs,
                                            const int _numFields,
                                            const cudaTextureObject_t *_compOps,
                                            const cudaTextureObject_t *_theta,
                                            const int _numOps,
                                            const ComposedFieldCuda *_compFields,
                                            const int _numCompFields,
                                            const int *_oneRingVerts,
                                            const float *_centroidWeights,
                                            const int *_oneRingScatterAddr,
                                            const float _sigma,
                                            const float _contactAngle);


__global__ void GenerateOneRingCentroidWeights_Kernel(glm::vec3 *d_verts,
                                                      const glm::vec3 *d_normals,
                                                      const int _numVerts,
                                                      float *_centroidWeights,
                                                      const int *_oneRingIds,
                                                      const glm::vec3 *_oneRingVerts,
                                                      const int *_numNeighsPerVert,
                                                      const int *_oneRingScatterAddr);


//---------------------------------------------------------------------------------------------------------------------------------

#endif //IMPLICITSKINKERNELS_H
