#include "ImplicitSkinKernels.h"


//------------------------------------------------------------------------------------------------
// CUDA Device Functions
//------------------------------------------------------------------------------------------------

__device__ void VertexProjection(glm::vec3 &_deformedVert,
                                 const float &_origIso,
                                 const float &_newIso,
                                 const glm::vec3 &_newIsoGrad,
                                 glm::vec3 &_prevIsoGrad,
                                 float &_gradAngle,
                                 const float &_sigma,
                                 const float &_contactAngle)
{
    float angle = _gradAngle = glm::angle(_newIsoGrad, _prevIsoGrad);
    if(angle < _contactAngle)
    {
        _deformedVert = _deformedVert + ( _sigma * (_newIso - _origIso) * (_newIsoGrad / glm::length2(_newIsoGrad)));
        _prevIsoGrad = _newIsoGrad;
    }
}

//------------------------------------------------------------------------------------------------

__device__ void TangentialRelaxation(glm::vec3 &_deformedVert,
                                     const glm::vec3 &_normal,
                                     const float &_origIso,
                                     const float &_newIso,
                                     const int *_oneRingNeigh,
                                     const float *_centroidWeights,
                                     const int _numNeighs,
                                     const glm::vec3 *_verts)
{
    float mu = 1.0f - (float)pow(fabs(_newIso- _origIso) - 1.0f, 4.0f);
    mu = mu < 0.0f ? 0.0f : mu;

    glm::vec3 sumWeightedCentroid(0.0f);

    for(int i=0; i<_numNeighs; i++)
    {
        glm::vec3 neighVert = _verts[_oneRingNeigh[i]];
        glm::vec3 projNeighVert = neighVert; // TODO project vert onto tangential plane using _normal
        float barycentricCoord = _centroidWeights[i];
        sumWeightedCentroid += barycentricCoord * projNeighVert;
    }

    _deformedVert = ((1.0f - mu) * _deformedVert) + (mu * sumWeightedCentroid);
}

//------------------------------------------------------------------------------------------------

__device__ void LaplacianSmoothing(glm::vec3 &_deformedVert,
                                   const glm::vec3 &_normal,
                                   const int *_oneRingNeigh,
                                   const float *_centroidWeights,
                                   const int _numNeighs,
                                   const glm::vec3 *_verts,
                                   const float _beta)
{
    glm::vec3 centroid(0.0f, 0.0f, 0.0f);

    for(int i=0; i<_numNeighs; i++)
    {
        centroid += (_centroidWeights[i] * _verts[_oneRingNeigh[i]]);
    }

    _deformedVert = ((1.0f - _beta) * _deformedVert) + (_beta * centroid);
}


//------------------------------------------------------------------------------------------------
// CUDA Global Kernels
//------------------------------------------------------------------------------------------------

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
        glm::vec3 transformedPoint = glm::vec3(_rigidTransforms[i] * glm::vec4(samplePoint,1.0f));
        glm::vec3 texturePoint = glm::vec3(textureSpace * glm::vec4(transformedPoint, 1.0f));

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

__global__ void SimpleEvaluateGlobalField(float *_output,
                                          glm::vec3 *_samplePoint,
                                          uint _numSamples,
                                          glm::mat4 *_textureSpace,
                                          glm::mat4 *_rigidTransforms,
                                          cudaTextureObject_t *_fieldFuncs,
                                          uint _numFields)
{
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    if(tid >= _numSamples)
    {
        return;
    }


    glm::vec3 samplePoint = _samplePoint[tid];


    float maxF = FLT_MIN;
    float f[100];

    for(int i=0; i<_numFields; i++)
    {
        glm::mat4 rigidTrans = _rigidTransforms[i];
        glm::mat4 textureSpace = _textureSpace[i];
//        glm::vec3 texturePoint = glm::vec3(textureSpace * glm::vec4(glm::vec3(glm::inverse(rigidTrans) * glm::vec4(samplePoint, 1.0f)), 1.0f));
        glm::vec3 texturePoint = glm::vec3(textureSpace * glm::inverse(rigidTrans) * glm::vec4(samplePoint, 1.0f));
//        texturePoint = ((((samplePoint / 800.0f)+glm::vec3(1.0f,1.0f,1.0f))*0.5f));

        f[i] = tex3D<float>(_fieldFuncs[i], (texturePoint.x*0.25f), texturePoint.y, texturePoint.z);

        maxF = (f[i]>maxF) ? f[i] : maxF;
    }

//    printf("%f\n", maxF);

    _output[tid] = maxF;
}

//------------------------------------------------------------------------------------------------

__global__ void LinearBlendWeightSkin_Kernel(glm::vec3 *_deformedVert,
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

    _deformedVert[tid] = glm::vec3(boneTransform * glm::vec4(_origVert[tid], 1.0f));

}

//------------------------------------------------------------------------------------------------

__global__ void ImplicitSkin_Kernel()
{

}


//------------------------------------------------------------------------------------------------
// Helper CPP Functions
//------------------------------------------------------------------------------------------------

uint kernels::iDivUp(uint a, uint b)
{
    uint c = a/b;
    c += (a%b == 0) ? 0: 1;
    return c;
}



//------------------------------------------------------------------------------------------------
// CPP functions
//------------------------------------------------------------------------------------------------

void kernels::LinearBlendWeightSkin(glm::vec3 *_deformedVert,
                           const glm::vec3 *_origVert,
                           const glm::mat4 *_transform,
                           const uint *_boneId,
                           const float *_weight,
                           const uint _numVerts,
                           const uint _numBones)
{
    uint numThreads = 1024u;
    uint numBlocks = kernels::iDivUp(_numVerts, numThreads);


    LinearBlendWeightSkin_Kernel<<<numBlocks, numThreads>>>(_deformedVert,
                                                            _origVert,
                                                            _transform,
                                                            _boneId,
                                                            _weight,
                                                            _numVerts,
                                                            _numBones);
    cudaThreadSynchronize();
}

//------------------------------------------------------------------------------------------------

void kernels::SimpleEval(float *_output,
                        glm::vec3 *_samplePoint,
                        uint _numSamples,
                        glm::mat4* _textureSpace,
                        glm::mat4 *_rigidTransforms,
                        cudaTextureObject_t *_fieldFuncs,
                        uint _numFields)
{
    uint numThreads = 1024u;
    uint numBlocks = kernels::iDivUp(_numSamples, numThreads);

    SimpleEvaluateGlobalField<<<numBlocks, numThreads>>>(_output,
                                                         _samplePoint,
                                                         _numSamples,
                                                         _textureSpace,
                                                         _rigidTransforms,
                                                         _fieldFuncs,
                                                         _numFields);
    cudaThreadSynchronize();
}
