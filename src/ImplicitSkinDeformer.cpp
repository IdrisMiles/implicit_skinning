#include "implicitskindeformer.h"
#include "ImplicitSkinKernels.h"
#include "helper_cuda.h"



//------------------------------------------------------------------------

ImplicitSkinDeformer::ImplicitSkinDeformer():
    m_initMeshCudaMem(false),
    m_initFieldCudaMem(false),
    m_initGobalFieldFunc(false),
    m_meshDeformedMapped(false)
{
    m_threads.resize(std::thread::hardware_concurrency()-1);
    checkCudaErrors(cudaSetDevice(0));
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

    if(m_initMeshCudaMem)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(m_meshVBO_CUDA));
        checkCudaErrors(cudaFree(d_meshOrigPtr));
        checkCudaErrors(cudaFree(d_transformPtr));
        checkCudaErrors(cudaFree(d_boneIdPtr));
        checkCudaErrors(cudaFree(d_weightPtr));
        m_initMeshCudaMem = false;
    }

    if(m_initFieldCudaMem)
    {
        checkCudaErrors(cudaFree(d_textureSpacePtr));
        checkCudaErrors(cudaFree(d_fieldsPtr));
    }
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::AttachMesh(const Mesh _origMesh,
                                      const GLuint _meshVBO,
                                      const std::vector<glm::mat4> &_transform)
{
    m_numVerts = _origMesh.m_meshVerts.size();
    InitMeshCudaMem(_origMesh, _meshVBO, _transform);
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

    int numThreads = m_threads.size() + 1;
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

    m_initGobalFieldFunc = true;

    InitFieldCudaMem();
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::PerformLBWSkinning(const std::vector<glm::mat4> &_transform)
{

    if(!m_initMeshCudaMem)
    {
        return;
    }

    // upload data
    checkCudaErrors(cudaMemcpy((void*)d_transformPtr, &_transform[0][0][0], _transform.size() * sizeof(glm::mat4), cudaMemcpyHostToDevice));


    kernels::LinearBlendWeightSkin(GetMeshDeformedDevicePtr(),
                                   d_meshOrigPtr,
                                   d_transformPtr,
                                   d_boneIdPtr,
                                   d_weightPtr,
                                   m_numVerts,
                                   _transform.size());


    cudaThreadSynchronize();
    getLastCudaError("LinearBlendWeightSkin Failed");


    // release cuda resources so openGL can render
    ReleaseMeshDeformedDevicePtr();
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::PerformImplicitSkinning(const std::vector<glm::mat4> &_transform)
{
    if(!m_initMeshCudaMem || !m_initFieldCudaMem)
    {
        return;
    }
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
void ImplicitSkinDeformer::SetRigidTransforms(const std::vector<glm::mat4> &_transforms)
{
    if(m_initGobalFieldFunc)
    {
        m_globalFieldFunction.SetRigidTransforms(_transforms);
    }

    if(m_initMeshCudaMem)
    {
        checkCudaErrors(cudaMemcpy((void*)d_transformPtr, &_transforms[0][0][0], _transforms.size() * sizeof(glm::mat4), cudaMemcpyHostToDevice));
    }
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::EvalField(std::vector<float> &_output, const std::vector<glm::vec3> &_samplePoints)
{
    _output.clear();
    if(!m_initGobalFieldFunc)
    {
        return;
    }

    _output.resize(_samplePoints.size());

    if(!m_initMeshCudaMem || !m_initFieldCudaMem)
    {
        EvalFieldCPU(_output, _samplePoints);
    }
    else
    {
        EvalFieldGPU(_output, _samplePoints);
    }
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::EvalFieldInCube(std::vector<float> &_output, const int dim, const float scale)
{
    _output.clear();
    if(!m_initGobalFieldFunc)
    {
        return;
    }

    _output.resize(dim *dim *dim);

    if(!m_initMeshCudaMem || !m_initFieldCudaMem)
    {
        EvalFieldInCubeCPU(_output, dim, scale);
    }
    else
    {
        EvalFieldInCubeGPU(_output, dim, scale);
    }
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::EvalFieldCPU(std::vector<float> &_output, const std::vector<glm::vec3> &_samplePoints)
{
    unsigned int numSamples = _samplePoints.size();

    int numThreads = m_threads.size() + 1;
    int chunkSize = numSamples / numThreads;
    int numBigChunks = numSamples % numThreads;
    int bigChunkSize = chunkSize + (numBigChunks>0 ? 1 : 0);
    int startChunk = 0;
    int threadId=0;


    auto threadFunc = [&, this](int startChunk, int endChunk){
        for(int i=startChunk;i<endChunk;i++)
        {
            _output[i] = m_globalFieldFunction.Eval(_samplePoints[i]);
        }
    };


    // Evalue field in each thread
    for(threadId=0; threadId<numBigChunks; threadId++)
    {
        m_threads[threadId] = std::thread(threadFunc, startChunk, (startChunk+bigChunkSize));
        startChunk+=bigChunkSize;
    }
    for(; threadId<numThreads-1; threadId++)
    {
        m_threads[threadId] = std::thread(threadFunc, startChunk, (startChunk+chunkSize));
        startChunk+=chunkSize;
    }
    threadFunc(startChunk, numSamples);

    for(int i=0; i<numThreads-1; i++)
    {
        if(m_threads[i].joinable())
        {
            m_threads[i].join();
        }
    }
}

//------------------------------------------------------------------------------------------------


void ImplicitSkinDeformer::EvalFieldGPU(std::vector<float> &_output, const std::vector<glm::vec3> &_samplePoints)
{
    uint numFields = m_globalFieldFunction.GetFieldFuncs().size();

    // allocate device memory
    float *d_output;
    glm::vec3 *d_samplePoints;
    checkCudaErrors(cudaMalloc(&d_output, _samplePoints.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_samplePoints, _samplePoints.size() * sizeof(glm::vec3)));

    // upload data to device
    checkCudaErrors(cudaMemcpy((void*)d_samplePoints, &_samplePoints[0], _samplePoints.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice));

    // run kernel
    kernels::SimpleEval(d_output, d_samplePoints, _samplePoints.size(), d_textureSpacePtr, d_transformPtr, d_fieldsPtr, numFields);
    getLastCudaError("Kernel::SimpleEval");

    // download data to host
    checkCudaErrors(cudaMemcpy(&_output[0], d_output, _samplePoints.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // free memory
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_samplePoints));
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::EvalFieldInCubeCPU(std::vector<float> &_output, const int dim, const float scale)
{

}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::EvalFieldInCubeGPU(std::vector<float> &_output, const int dim, const float scale)
{

}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::InitMeshCudaMem(const Mesh _origMesh,
                                           const GLuint _meshVBO,
                                           const std::vector<glm::mat4> &_transform)
{

    if(m_initMeshCudaMem) { return; }

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

    m_initMeshCudaMem = true;
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::InitFieldCudaMem()
{
    if(!m_initGobalFieldFunc || m_initFieldCudaMem) { return; }


    auto fieldFuncs = m_globalFieldFunction.GetFieldFuncs();
    auto compOps = m_globalFieldFunction.GetCompOps();
    auto compFields = m_globalFieldFunction.GetCompFields();

    // create device memory here
    uint numFields = fieldFuncs.size();
    uint numCompOps = compOps.size();
    uint numCompFields = compFields.size();

    // allocate memory
    checkCudaErrors(cudaMalloc(&d_textureSpacePtr, numFields * sizeof(glm::mat4)));
    checkCudaErrors(cudaMalloc(&d_fieldsPtr, numFields * sizeof(cudaTextureObject_t)));
    checkCudaErrors(cudaMalloc(&d_compOpPtr, numCompOps * sizeof(cudaTextureObject_t)));
    checkCudaErrors(cudaMalloc(&d_compFieldPtr, numCompFields * sizeof(ComposedFieldCuda)));

    // upload data
    std::vector<glm::mat4> texSpaceTrans(numFields);
    for(int i=0; i<numFields; ++i)
    {
        texSpaceTrans[i] = fieldFuncs[i]->GetTextureSpaceTransform();
        auto fieldTex = fieldFuncs[i]->GetFieldFunc3DTexture();
        cudaMemcpy(d_fieldsPtr+i, &fieldTex, 1*sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    }
    for(int i=0; i<numCompOps; ++i)
    {
        auto compOpTex = compOps[i]->GetFieldFunc3DTexture();
        cudaMemcpy(d_compOpPtr+i, &compOpTex, 1*sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    }
    for(int i=0; i<numCompFields; ++i)
    {
        auto cf = compFields[i];
        cudaMemcpy(d_compFieldPtr+i, &cf, 1*sizeof(ComposedFieldCuda), cudaMemcpyHostToDevice);
    }
    checkCudaErrors(cudaMemcpy((void*)d_textureSpacePtr, &texSpaceTrans[0][0][0], texSpaceTrans.size() * sizeof(glm::mat4), cudaMemcpyHostToDevice));

    m_initFieldCudaMem = true;
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
