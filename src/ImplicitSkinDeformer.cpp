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
    m_threads.resize(std::thread::hardware_concurrency()-1);
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
    for(int i=0; i<m_threads.size(); i++)
    {
        if(m_threads[i].joinable())
        {
            m_threads[i].join();
        }
    }

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

    kernels::LinearBlendWeightSkin(GetMeshDeformedDevicePtr(),
                           d_meshOrigPtr,
                           d_transformPtr,
                           d_boneIdPtr,
                           d_weightPtr,
                           m_numVerts,
                           _transform.size());

    getLastCudaError("LinearBlendWeightSkin Failed");

    checkCudaErrors(cudaThreadSynchronize());
    ReleaseMeshDeformedDevicePtr();
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::PerformImplicitSkinning(const std::vector<glm::mat4> &_transform)
{

}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::AddComposedField(std::shared_ptr<ComposedField> _composedField)
{
    m_globalFieldFunction.AddComposedField(_composedField);
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::AddFieldFunction(std::shared_ptr<FieldFunction> _fieldFunc)
{
    m_globalFieldFunction.AddFieldFunction(_fieldFunc);
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::AddCompositionOp(std::shared_ptr<CompositionOp> _compOp)
{
    m_globalFieldFunction.AddCompositionOp(_compOp);
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::GenerateGlobalFieldFunction(const std::vector<Mesh> &_meshParts,
                                                       const std::vector<glm::vec3> &_boneStarts,
                                                       const std::vector<glm::vec3> &_boneEnds,
                                                       const int _numHrbfCentres)
{
    m_globalFieldFunction.Fit(_meshParts.size());

    auto threadFunc = [&, this](int startId, int endId){
        for(int mp=startId; mp<endId; mp++)
        {
            Mesh hrbfCentres;
            m_globalFieldFunction.GenerateHRBFCentres(_meshParts[mp], _boneStarts[mp], _boneEnds[mp], _numHrbfCentres, hrbfCentres);
            m_globalFieldFunction.GenerateFieldFuncs(hrbfCentres, _meshParts[mp], mp);

        }
    };

    int numThreads = m_threads.size() + 1;// std::thread::hardware_concurrency();
    int dataSize = _meshParts.size();
    int chunkSize = dataSize / numThreads;
    int numBigChunks = dataSize % numThreads;
    int bigChunkSize = chunkSize + (numBigChunks>0 ? 1 : 0);
    int startChunk = 0;
    int threadId=0;

    // Generate Field functions in each thread
    for(threadId=0; threadId<numBigChunks; threadId++)
    {
        m_threads[threadId] = std::thread(threadFunc, startChunk, startChunk+bigChunkSize);
        startChunk+=bigChunkSize;
    }
    for(; threadId<numThreads-1; threadId++)
    {
        m_threads[threadId] = std::thread(threadFunc, startChunk, startChunk+chunkSize);
        startChunk+=chunkSize;
    }
    threadFunc(startChunk, _meshParts.size());

    for(int i=0; i<numThreads-1; i++)
    {
        if(m_threads[i].joinable())
        {
            m_threads[i].join();
        }
    }


    m_globalFieldFunction.GenerateGlobalFieldFunc();
}

//------------------------------------------------------------------------------------------------
void ImplicitSkinDeformer::SetRigidTransforms(const std::vector<glm::mat4> &_transforms)
{
    m_globalFieldFunction.SetRigidTransforms(_transforms);
    checkCudaErrors(cudaMemcpy((void*)d_transformPtr, &_transforms[0][0][0], _transforms.size() * sizeof(glm::mat4), cudaMemcpyHostToDevice));
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::SimpleEval(std::vector<float> &_output,
                                      const std::vector<glm::vec3> &_samplePoints,
                                      const std::vector<glm::mat4> &_transform)
{
    // create device memory here
    float *d_output;
    glm::vec3 *d_samplePoints;
    glm::mat4 *d_textureSpace;
    cudaTextureObject_t *d_fields;
    uint numFields;
    checkCudaErrors(cudaMalloc(&d_output, _samplePoints.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_samplePoints, _samplePoints.size() * sizeof(glm::vec3)));
    checkCudaErrors(cudaMemcpy((void*)d_samplePoints, &_samplePoints[0], _samplePoints.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void*)d_transformPtr, &_transform[0][0][0], _transform.size() * sizeof(glm::mat4), cudaMemcpyHostToDevice));


    // run kernel
    kernels::SimpleEval(d_output, d_samplePoints, _samplePoints.size(), d_textureSpace, d_transformPtr, d_fields, numFields);

    // copy results to host memory
    _output.clear();
    _output.resize(_samplePoints.size());
    checkCudaErrors(cudaMemcpy(&_output[0], d_output, _samplePoints.size() * sizeof(float), cudaMemcpyDeviceToHost));
}

//------------------------------------------------------------------------------------------------

GlobalFieldFunction &ImplicitSkinDeformer::GetGlocalFieldFunc()
{
    return m_globalFieldFunction;
}


//------------------------------------------------------------------------------------------------

glm::vec3 *ImplicitSkinDeformer::GetMeshDeformedDevicePtr()
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

void ImplicitSkinDeformer::ReleaseMeshDeformedDevicePtr()
{
    if(m_meshDeformedMapped)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &m_meshVBO_CUDA, 0));
        m_meshDeformedMapped = false;
    }
}
