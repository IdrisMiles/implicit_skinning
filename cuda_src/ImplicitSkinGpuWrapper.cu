#include "ImplicitSkinGpuWrapper.h"
#include "ImplicitSkinKernels.h"
#include <cuda_runtime.h>
#include <stdio.h>



//------------------------------------------------------------------------------------------------

uint isgw::iDivUp(uint a, uint b)
{
    uint c = a/b;
    c += (a%b == 0) ? 0: 1;
    return c;
}

//------------------------------------------------------------------------------------------------

void isgw::LinearBlendWeightSkin(glm::vec3 *_deformedVert,
                                   const glm::vec3 *_origVert,
                                    glm::vec3 *_deformedNorms,
                                    const glm::vec3 *_origNorms,
                                   const glm::mat4 *_transform,
                                   const uint *_boneId,
                                   const float *_weight,
                                   const int _numVerts,
                                   const int _numBones)
{
    uint numThreads = 1024u;
    uint numBlocks = isgw::iDivUp(_numVerts, numThreads);


    LinearBlendWeightSkin_Kernel<<<numBlocks, numThreads>>>(_deformedVert,
                                                            _origVert,
                                                            _deformedNorms,
                                                            _origNorms,
                                                            _transform,
                                                            _boneId,
                                                            _weight,
                                                            _numVerts,
                                                            _numBones);
    cudaThreadSynchronize();
}

//------------------------------------------------------------------------------------------------

void isgw::EvalGlobalField(float *_output,
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
                              const int _numCompFields)
{
    uint numThreads = 1024u;
    uint numBlocks = isgw::iDivUp(_numSamples, numThreads);

    EvalGlobalField_Kernel<<<numBlocks, numThreads>>>(_output, _samplePoint, _numSamples,
                                                          _textureSpace, _rigidTransforms, _fieldFuncs, _numFields,
                                                          _compOps, _theta, _numOps,
                                                          _compFields, _numCompFields);

    cudaThreadSynchronize();
}

//------------------------------------------------------------------------------------------------

void isgw::EvalGradGlobalField(float *_output,
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
                              const int _numCompFields)
{
    uint numThreads = 1024u;
    uint numBlocks = isgw::iDivUp(_numSamples, numThreads);

    printf("%u\n",numBlocks);

    EvalGradGlobalField_Kernel<<<numBlocks, numThreads>>>(_output, _outputG, _samplePoint, _numSamples,
                                                          _textureSpace, _rigidTransforms, _fieldFuncs, _numFields,
                                                          _compOps, _theta, _numOps,
                                                          _compFields, _numCompFields);

    cudaThreadSynchronize();
}

//------------------------------------------------------------------------------------------------

void isgw::SimpleImplicitSkin(glm::vec3 *_deformedVert,
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
                              const int _iterations)
{
    uint numThreads = 1024u;
    uint numBlocks = isgw::iDivUp(_numVerts, numThreads);

    for(int i=0; i<_iterations; i++)
    {
        VertexProjection_Kernel<<<numBlocks, numThreads>>>(_deformedVert, _normal, _origIsoValue, _prevIsoGrad, _numVerts,
                                                             _textureSpace, _rigidTransforms, _fieldFuncs, _numFields,
                                                           _compOps, _theta, _numOps,
                                                           _compFields, _numCompFields,
                                                           _sigma, _contactAngle);

        cudaThreadSynchronize();

        TangentialRelaxation_Kernel<<<numBlocks, numThreads>>>(_deformedVert, _normal, _origIsoValue, _prevIsoGrad, _numVerts,
                                                               _textureSpace, _rigidTransforms, _fieldFuncs, _numFields,
                                                               _compOps, _theta, _numOps,
                                                               _compFields, _numCompFields,
                                                               _oneRingVerts, _centroidWeights, _neighScatterAddr,
                                                               _sigma, _contactAngle);

        cudaThreadSynchronize();
    }

}

//------------------------------------------------------------------------------------------------

void isgw::GenerateScatterAddress(int *begin,
                                     int *end,
                                     int *scatteredAddr)
{
    thrust::device_ptr<int> beginPtr = thrust::device_pointer_cast(begin);
    thrust::device_ptr<int> endPtr = thrust::device_pointer_cast(end);
    thrust::device_ptr<int> scatteredAddrPtr = thrust::device_pointer_cast(scatteredAddr);
    thrust::exclusive_scan(beginPtr, endPtr, scatteredAddrPtr);
}

//------------------------------------------------------------------------------------------------

void isgw::GenerateOneRingCentroidWeights(glm::vec3 *d_verts,
                                             const glm::vec3 *d_normals,
                                             const int _numVerts,
                                             float *_centroidWeights,
                                             const int *_oneRingIds,
                                             const glm::vec3 *_oneRingVerts,
                                             const int *_numNeighsPerVert,
                                             const int *_oneRingScatterAddr)
{
    uint numThreads = 1024u;
    uint numBlocks = isgw::iDivUp(_numVerts, numThreads);

    GenerateOneRingCentroidWeights_Kernel<<<numBlocks, numThreads>>>(d_verts,
                                                                     d_normals,
                                                                     _numVerts,
                                                                     _centroidWeights,
                                                                     _oneRingIds,
                                                                     _oneRingVerts,
                                                                     _numNeighsPerVert,
                                                                     _oneRingScatterAddr);

    cudaThreadSynchronize();
}

//------------------------------------------------------------------------------------------------
