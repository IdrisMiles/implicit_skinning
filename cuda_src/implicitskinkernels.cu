#include "implicitskinkernels.h"
#include "cutil_math.h"

#include "ScalarField/compfield.h"
#include <stdio.h>



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
//    if(totalWeight < 0.8f)
//    {
//        printf("v: %i, w: %f\n",tid, totalWeight);
//    }


    _deformedVert[tid] = boneTransform * glm::vec4(_origVert[tid], 1.0f);

}


//------------------------------------------------------------------------

uint iDivUp(uint a, uint b)
{
    uint c = a/b;
    c += (a%b == 0) ? 0: 1;
    return c;
}

//------------------------------------------------------------------------

ImplicitSkinKernels::ImplicitSkinKernels(const Mesh _origMesh,
                                         const GLuint _meshVBO,
                                         const std::vector<glm::mat4> &_transform)
{
    m_numVerts = _origMesh.m_meshVerts.size();

    // Get bone ID and weights per vertex
    unsigned int boneIds[m_numVerts *4];
    float weights[m_numVerts *4];
    int i=0;
    for(auto &bw : _origMesh.m_meshBoneWeights)
    {
        float totalW = 0.0f;
        for(int j=0; j<4; j++)
        {
            boneIds[i+j] = bw.boneID[j];
            weights[i+j] = bw.boneWeight[j];
            totalW += bw.boneWeight[j];
        }

        // Normalize weights
        if(totalW < 1.0f)
        {
            for(int j=0; j<4; j++)
            {
                weights[i+j] /= totalW;
            }
        }
        i+=4;
    }


    cudaSetDevice(0);


    // Register vertex buffer with CUDA
    cudaGraphicsGLRegisterBuffer(&m_meshVBO_CUDA, _meshVBO, cudaGraphicsMapFlagsWriteDiscard);

    // Allocate cuda memory
    cudaMalloc(&d_meshOrigPtr, m_numVerts * sizeof(glm::vec3));
    cudaMalloc(&d_transformPtr, _transform.size() * sizeof(glm::mat4));
    cudaMalloc(&d_boneIdPtr, m_numVerts * 4 * sizeof(unsigned int));
    cudaMalloc(&d_weightPtr, m_numVerts * 4 * sizeof(float));

    // copy memory over to cuda
    cudaMemcpy((void*)d_meshOrigPtr, (void*)&_origMesh.m_meshVerts[0], m_numVerts * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_transformPtr, (void*)&_transform[0][0][0], _transform.size() * sizeof(glm::mat4), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_boneIdPtr, (void*)boneIds, m_numVerts *4* sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_weightPtr, (void*)weights, m_numVerts *4* sizeof(float), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
}

ImplicitSkinKernels::~ImplicitSkinKernels()
{

    cudaGraphicsUnregisterResource(m_meshVBO_CUDA);
    cudaFree(d_meshOrigPtr);
    cudaFree(d_transformPtr);
    cudaFree(d_boneIdPtr);
    cudaFree(d_weightPtr);
}


void ImplicitSkinKernels::PerformLBWSkinning(const std::vector<glm::mat4> &_transform)
{
    uint numThreads = 1024u;
    uint numBlocks = iDivUp(m_numVerts, numThreads);

    cudaMemcpy((void*)d_transformPtr, &_transform[0][0][0], _transform.size() * sizeof(glm::mat4), cudaMemcpyHostToDevice);

    LinearBlendWeightSkin<<<numBlocks, numThreads>>>(GetMeshDeformedPtr(),
                                                     d_meshOrigPtr,
                                                     d_transformPtr,
                                                     d_boneIdPtr,
                                                     d_weightPtr,
                                                     m_numVerts,
                                                     _transform.size());

    cudaThreadSynchronize();
    ReleaseMeshDeformedPtr();
}



glm::vec3 *ImplicitSkinKernels::GetMeshDeformedPtr()
{
    if(!m_meshDeformedMapped)
    {
        size_t numBytes;
        cudaGraphicsMapResources(1, &m_meshVBO_CUDA, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&d_meshDeformedPtr, &numBytes, m_meshVBO_CUDA);

        m_meshDeformedMapped = true;
    }

    return d_meshDeformedPtr;
}

void ImplicitSkinKernels::ReleaseMeshDeformedPtr()
{
    if(m_meshDeformedMapped)
    {
        cudaGraphicsUnmapResources(1, &m_meshVBO_CUDA, 0);
        m_meshDeformedMapped = false;
    }
}
