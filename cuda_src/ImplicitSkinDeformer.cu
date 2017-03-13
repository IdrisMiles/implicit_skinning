#include "implicitskindeformer.h"
#include "ImplicitSkinKernels.h"
#include "helper_cuda.h"



//------------------------------------------------------------------------

ImplicitSkinDeformer::ImplicitSkinDeformer(const Mesh _origMesh,
                                         const GLuint _meshVBO,
                                         const std::vector<glm::mat4> &_transform):
    m_init(false),
    m_meshDeformedMapped(false)
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


    if(!m_init)
    {
        checkCudaErrors(cudaSetDevice(0));

        // Register vertex buffer with CUDA
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_meshVBO_CUDA, _meshVBO, cudaGraphicsMapFlagsWriteDiscard));

        // Allocate cuda memory
        checkCudaErrorsMsg(cudaMalloc(&d_meshOrigPtr, m_numVerts * sizeof(glm::vec3)), "Allocate memory for original mesh");
        checkCudaErrorsMsg(cudaMalloc(&d_transformPtr, _transform.size() * sizeof(glm::mat4)), "Allocate memory for transforms");
        checkCudaErrorsMsg(cudaMalloc(&d_boneIdPtr, m_numVerts * 4 * sizeof(unsigned int)), "Allocate memory for bone Ids");
        checkCudaErrorsMsg(cudaMalloc(&d_weightPtr, m_numVerts * 4 * sizeof(float)), "Allocate memory for bone weights");

        // copy memory over to cuda
        checkCudaErrors(cudaMemcpy((void*)d_meshOrigPtr, (void*)&_origMesh.m_meshVerts[0], m_numVerts * sizeof(glm::vec3), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void*)d_transformPtr, (void*)&_transform[0][0][0], _transform.size() * sizeof(glm::mat4), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void*)d_boneIdPtr, (void*)boneIds, m_numVerts *4* sizeof(unsigned int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void*)d_weightPtr, (void*)weights, m_numVerts *4* sizeof(float), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaThreadSynchronize());
    }
    m_init = true;
}

//------------------------------------------------------------------------------------------------

ImplicitSkinDeformer::~ImplicitSkinDeformer()
{

    if(m_init)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(m_meshVBO_CUDA));
        checkCudaErrors(cudaFree(d_meshOrigPtr));
        checkCudaErrors(cudaFree(d_transformPtr));
        checkCudaErrors(cudaFree(d_boneIdPtr));
        checkCudaErrors(cudaFree(d_weightPtr));
        m_init = false;
    }
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::PerformLBWSkinning(const std::vector<glm::mat4> &_transform)
{

    if(!m_init)
    {
        return;
    }

    checkCudaErrors(cudaMemcpy((void*)d_transformPtr, &_transform[0][0][0], _transform.size() * sizeof(glm::mat4), cudaMemcpyHostToDevice));

    LinearBlendWeightSkin(GetMeshDeformedPtr(),
                           d_meshOrigPtr,
                           d_transformPtr,
                           d_boneIdPtr,
                           d_weightPtr,
                           m_numVerts,
                           _transform.size());

    getLastCudaError("LinearBlendWeightSkin Failed");

    checkCudaErrors(cudaThreadSynchronize());
    ReleaseMeshDeformedPtr();
}

//------------------------------------------------------------------------------------------------

glm::vec3 *ImplicitSkinDeformer::GetMeshDeformedPtr()
{
    if(!m_meshDeformedMapped)
    {
        size_t numBytes;
        checkCudaErrors(cudaGraphicsMapResources(1, &m_meshVBO_CUDA, 0));
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_meshDeformedPtr, &numBytes, m_meshVBO_CUDA));

        m_meshDeformedMapped = true;
    }

    return d_meshDeformedPtr;
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::ReleaseMeshDeformedPtr()
{
    if(m_meshDeformedMapped)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &m_meshVBO_CUDA, 0));
        m_meshDeformedMapped = false;
    }
}
