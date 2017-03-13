#include "implicitskindeformer.h"
#include "ImplicitSkinKernels.cuh"


//------------------------------------------------------------------------

uint iDivUp(uint a, uint b)
{
    uint c = a/b;
    c += (a%b == 0) ? 0: 1;
    return c;
}

//------------------------------------------------------------------------

ImplicitSkinDeformer::ImplicitSkinDeformer(const Mesh _origMesh,
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

//------------------------------------------------------------------------------------------------

ImplicitSkinDeformer::~ImplicitSkinDeformer()
{

    cudaGraphicsUnregisterResource(m_meshVBO_CUDA);
    cudaFree(d_meshOrigPtr);
    cudaFree(d_transformPtr);
    cudaFree(d_boneIdPtr);
    cudaFree(d_weightPtr);
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::PerformLBWSkinning(const std::vector<glm::mat4> &_transform)
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

//------------------------------------------------------------------------------------------------

glm::vec3 *ImplicitSkinDeformer::GetMeshDeformedPtr()
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

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::ReleaseMeshDeformedPtr()
{
    if(m_meshDeformedMapped)
    {
        cudaGraphicsUnmapResources(1, &m_meshVBO_CUDA, 0);
        m_meshDeformedMapped = false;
    }
}
