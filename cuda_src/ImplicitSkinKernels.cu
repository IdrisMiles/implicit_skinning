#include "ImplicitSkinKernels.cuh"


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
                                    uint _numCompFields)
{
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    if(tid >= _numSamples)
    {
        return;
    }


    glm::vec3 samplePoint = _samplePoint[tid];
    glm::mat4 textureSpace = _textureSpace;


    float maxF = FLT_MIN;
    float f[100];
//    float3 df[100];
    int i=0;
    for(i=0; i<_numFields; i++)
    {
        glm::vec3 transformedPoint = _rigidTransforms[i] * glm::vec4(samplePoint,1.0f);
        glm::vec3 texturePoint = textureSpace * glm::vec4(transformedPoint, 1.0f);

        f[i] = tex3D<float>(_fieldFuncs[i], texturePoint.x, texturePoint.y, texturePoint.z);
//        df[i] = tex3D<float3>(_fieldDeriv[i], texturePoint.x, texturePoint.y, texturePoint.z);

        maxF = (f[i]>maxF) ? f[i] : maxF;
    }


//    float cf[100];
//    float maxF = FLT_MIN;
//    for(i=0; i<_numCompFields; i++)
//    {
//        int f1Id = _compFields[i].fieldFuncA;
//        int f2Id = _compFields[i].fieldFuncB;
//        int coId = _compFields[i].compOp;

//        glm::vec3 df1(df[f1Id].x, df[f1Id].y, df[f1Id].z);
//        glm::vec3 df2(df[f2Id].x, df[f2Id].y, df[f2Id].z);
//        float angle = glm::angle(df1, df2);
//        float theta = tex1D<float>(_theta, angle*0.5f*M_1_PI);

//        cf[i] = tex3D<float>(_compOps[coId], f[f1Id], f[f2Id], theta);

//        maxF = (cf[i]>maxF) ? cf[i] : maxF;
//    }


    _output[tid] = maxF;
}

//------------------------------------------------------------------------------------------------

__global__ void LinearBlendWeightSkin(glm::vec3 *_deformedVert,
                                      const glm::vec3 *_origVert,
                                      const glm::mat4 *_transform,
                                      const uint *_boneId,
                                      const float *_weight,
                                      const uint _numVerts,
                                      const uint _numBones)
{
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    if(tid >= _numVerts)
    {
        return;
    }

    glm::mat4 boneTransform = glm::mat4(0.0f);

    float totalWeight = 0.0f;
    for(int i=0; i<4; i++)
    {
        unsigned int boneId = _boneId[(tid*4) + i];
        float w = _weight[(tid*4) + i];
        boneTransform += (_transform[boneId] * w);

        totalWeight+=w;
    }

    _deformedVert[tid] = boneTransform * glm::vec4(_origVert[tid], 1.0f);

}

