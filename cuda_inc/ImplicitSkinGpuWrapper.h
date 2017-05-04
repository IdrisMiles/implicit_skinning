#ifndef IMPLICITSKINGPUWRAPPER_H
#define IMPLICITSKINGPUWRAPPER_H

//--------------------------------------------------------------------------------------------------------------

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


//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @data 18/04/2017
//--------------------------------------------------------------------------------------------------------------

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
                           const int _numVerts,
                           const int _numBones);


/// @brief Function to launch CUDA Kernel to perform implicit skinning
/// @param _deformedVert : Device pointer to the deformed vertices, this is the pointer that is mapped to the vertex buffer object
/// @param _normal : Device pointer to the deformed normals, this is the pointer that is mapped to the normals buffer object
/// @param _origIsoValue : Device pointer to iso values of each vertex in their original rest position.
/// @param _prevIsoGrad : Device pointer to the gradient of the field for each vertex from the previous frame.
/// @param _numVerts: number of vertices.
/// @param _textureSpace : Device pointer to the texture space transformation required to transform the sample space into normalise texture space
/// @param _rigidTransforms : Device pointer to the inverse bone transforms, required to transform sample space before sampling transformed fields.
/// @param _fieldFuncs : Device pointer to field function 3D textures.
/// @param  _numFields : Number of primitive fields.
/// @param  _compOps : Device pointer to to composition operator 3D textures.
/// @param _theta : Device pointer to theta opening function 1D texture.
/// @param _numOps : Number of composition operators.
/// @param _compFields : Device pointer to composed fields, a structure that holds ids for primitive fields and their composition operator
/// @param _numCompFields : Number of composed fields
/// @param _oneRingVerts : Device pointer to flat array of vertex ids of each vertices one ring nighbour.
/// @param _centroidWeights : Device pointer to one ring centroid weights calculated using MVC.
/// @param _neighScatterAddr : Device pointer holding start index into _oneRingVerts for each vertex.
/// @param _sigma : scaling value for vertex projection step.
/// @param _contactAngle : threshold for running vertex projection step
/// @param _iterations : max number of iterations of vertex projection and tengential relaxations.
void SimpleImplicitSkin(glm::vec3 *_deformedVert,
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
                        const int *_neighScatterAddr,
                        const float _sigma,
                        const float _contactAngle,
                        const int _iterations);


/// @brief Function to launch CUDA Kernel to evaluate the global field
/// @param _output : Device pointer to the output
/// @param _samplePoint : Const device pointer to the sample points
/// @param _numSamples: The number of samples
/// @param _textureSpace : Const device pointer to the texture space transform matrices, transform world space to texture space
/// @param _rigidTransform : Const device pointer to the rigid bone transform matrices
/// @param _fieldFuncs : Const device pointer to an array of textures holding the individual field functions
/// @param _numFields : The number of field function textures
/// @param  _compOps : Device pointer to to composition operator 3D textures.
/// @param _theta : Device pointer to theta opening function 1D texture.
/// @param _numOps : Number of composition operators.
/// @param _compFields : Device pointer to composed fields, a structure that holds ids for primitive fields and their composition operator
/// @param _numCompFields : Number of composed fields
void EvalGlobalField(float *_output,
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


/// @brief Function to launch CUDA Kernel to evaluate gradient of global field
/// @param _output : Device pointer to the field output
/// @param _outputG : Device pointer to gradient output
/// @param _samplePoint : Const device pointer to the sample points
/// @param _numSamples: The number of samples
/// @param _textureSpace : Const device pointer to the texture space transform matrices, transform world space to texture space
/// @param _rigidTransform : Const device pointer to the rigid bone transform matrices
/// @param _fieldFuncs : Const device pointer to an array of textures holding the individual field functions
/// @param _numFields : The number of field function textures
/// @param  _compOps : Device pointer to to composition operator 3D textures.
/// @param _theta : Device pointer to theta opening function 1D texture.
/// @param _numOps : Number of composition operators.
/// @param _compFields : Device pointer to composed fields, a structure that holds ids for primitive fields and their composition operator
/// @param _numCompFields : Number of composed fields
void EvalGradGlobalField(float *_output,
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


/// @brief Method to generate scatter address, runs an exclusive scan
void GenerateScatterAddress(int *begin,
                            int *end,
                            int *scatteredAddr);


/// @brief Method to generate the one ring centroid weights for each mesh vertex using Mean Value Coordinates method
/// @param _verts : Device pointer to vertices
/// @param _normals : Device pointer to normals
/// @param _numVerts : Numbe rof vertices
/// @param _centroidWeights :
void GenerateOneRingCentroidWeights(glm::vec3 *_verts,
                                    const glm::vec3 *_normals,
                                    const int _numVerts,
                                    float *_centroidWeights,
                                    const int *_oneRingIds,
                                    const glm::vec3 *_oneRingVerts,
                                    const int *_numNeighsPerVert,
                                    const int *_oneRingScatterAddr);


}

//--------------------------------------------------------------------------------------------------------------

#endif //IMPLICITSKINGPUWRAPPER_H
