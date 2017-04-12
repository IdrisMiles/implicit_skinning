#include "implicitskindeformer.h"
#include "ImplicitSkinKernels.h"
#include "helper_cuda.h"


//------------------------------------------------------------------------

ImplicitSkinDeformer::ImplicitSkinDeformer():
    m_initMeshCudaMem(false),
    m_initFieldCudaMem(false),
    m_initGobalFieldFunc(false),
    m_deformedMeshVertsMapped(false),
    m_deformedMeshNormsMapped(false)
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

    DestroyMeshCudaMem();

    DestroyFieldCudaMem();
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::AttachMesh(const Mesh _origMesh,
                                      const GLuint _meshVBO,
                                      const GLuint _meshNBO,
                                      const std::vector<glm::mat4> &_transform)
{
    InitMeshCudaMem(_origMesh, _meshVBO, _meshNBO, _transform);
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::GenerateGlobalFieldFunction(const std::vector<Mesh> &_meshParts,
                                                       const std::vector<glm::vec3> &_boneStarts,
                                                       const std::vector<glm::vec3> &_boneEnds,
                                                       const int _numHrbfCentres)
{
    m_globalFieldFunction.Fit(_meshParts.size());

    // Generate individual field functions per mesh part
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

    // Generate global field function
    m_globalFieldFunction.GenerateGlobalFieldFunc();
    m_initGobalFieldFunc = true;

    InitFieldCudaMem();

    //---------------------------------------
    // TODO
    // Initialise iso values for verts here

}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::Deform()
{
    PerformLBWSkinning();
//    PerformImplicitSkinning();
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::PerformLBWSkinning()
{

    if(!m_initMeshCudaMem)
    {
        return;
    }


    kernels::LinearBlendWeightSkin(GetDeformedMeshVertsDevicePtr(),
                                   d_origMeshVertsPtr,
                                   GetDeformedMeshNormsDevicePtr(),
                                   d_origMeshNormsPtr,
                                   d_transformPtr,
                                   d_boneIdPtr,
                                   d_weightPtr,
                                   m_numVerts,
                                   m_numTransforms);


    cudaThreadSynchronize();
    getLastCudaError("LinearBlendWeightSkin Failed");


    // release cuda resources so openGL can render
    ReleaseDeformedMeshVertsDevicePtr();
    ReleaseDeformedMeshNormsDevicePtr();
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::PerformImplicitSkinning()
{
    if(!m_initMeshCudaMem || !m_initFieldCudaMem)
    {
        return;
    }


    kernels::SimpleImplicitSkin(GetDeformedMeshVertsDevicePtr(),
                                GetDeformedMeshNormsDevicePtr(),
                                d_origVertIsoPtr,
                                m_numVerts,
                                d_textureSpacePtr,
                                d_transformPtr,
                                d_fieldsPtr,
                                d_gradPtr,
                                m_numFields,
                                d_oneRingIdPtr,
                                d_centroidWeightsPtr,
                                d_numNeighsPerVertPtr,
                                d_oneRingScatterAddrPtr);



    cudaThreadSynchronize();
    getLastCudaError("Implicit Skinning Failed");


    // release cuda resources so openGL can render
    ReleaseDeformedMeshVertsDevicePtr();
    ReleaseDeformedMeshNormsDevicePtr();
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
        m_numTransforms = _transforms.size();
        checkCudaErrors(cudaMemcpy((void*)d_transformPtr, &_transforms[0][0][0], m_numTransforms * sizeof(glm::mat4), cudaMemcpyHostToDevice));
    }
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::EvalGlobalField(std::vector<float> &_output, const std::vector<glm::vec3> &_samplePoints)
{
    _output.clear();
    if(!m_initGobalFieldFunc)
    {
        return;
    }

    _output.resize(_samplePoints.size());

    if(!m_initMeshCudaMem || !m_initFieldCudaMem)
    {
        EvalGlobalFieldCPU(_output, _samplePoints);
    }
    else
    {
        EvalGlobalFieldGPU(_output, _samplePoints);
    }
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::EvalGlobalFieldInCube(std::vector<float> &_output, const int dim, const float scale)
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

void ImplicitSkinDeformer::EvalGlobalFieldCPU(std::vector<float> &_output, const std::vector<glm::vec3> &_samplePoints)
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


void ImplicitSkinDeformer::EvalGlobalFieldGPU(std::vector<float> &_output, const std::vector<glm::vec3> &_samplePoints)
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
    kernels::SimpleEvalGlobalField(d_output, d_samplePoints, _samplePoints.size(), d_textureSpacePtr, d_transformPtr, d_fieldsPtr, numFields);
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
                                           const GLuint _meshNBO,
                                           const std::vector<glm::mat4> &_transform)
{

    if(m_initMeshCudaMem) { return; }

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

    // Get one ring neighbourhood
    std::vector<std::vector<int>> oneRing;
    std::vector<int> oneRingIdFlat;
    std::vector<glm::vec3> oneRingVertFlat;
    std::vector<int>numNeighsPerVertex;
    _origMesh.GetOneRingNeighours(oneRing);

    for(auto &neighList : oneRing)
    {
        numNeighsPerVertex.push_back(neighList.size());

        oneRingIdFlat.insert(oneRingIdFlat.begin(), neighList.begin(), neighList.end());

        for(auto &neighId : neighList)
        {
            oneRingVertFlat.push_back(_origMesh.m_meshVerts[neighId]);
        }
    }


    // Register vertex buffer with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_meshVBO_CUDA, _meshVBO, cudaGraphicsMapFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_meshNBO_CUDA, _meshNBO, cudaGraphicsMapFlagsWriteDiscard));


    // Allocate cuda memory
    checkCudaErrorsMsg(cudaMalloc(&d_origMeshVertsPtr,  m_numVerts * sizeof(glm::vec3)),          "Allocate memory for original mesh verts");
    checkCudaErrorsMsg(cudaMalloc(&d_origMeshNormsPtr,  m_numVerts * sizeof(glm::vec3)),          "Allocate memory for original mesh normals");
    checkCudaErrorsMsg(cudaMalloc(&d_origVertIsoPtr,    m_numVerts * sizeof(float)),          "Allocate memory for original vert iso values");
    checkCudaErrorsMsg(cudaMalloc(&d_newVertIsoPtr,    m_numVerts * sizeof(float)),          "Allocate memory for new vert iso values");
    checkCudaErrorsMsg(cudaMalloc(&d_transformPtr,      _transform.size() * sizeof(glm::mat4)),  "Allocate memory for transforms");
    checkCudaErrorsMsg(cudaMalloc(&d_boneIdPtr,         m_numVerts * 4 * sizeof(unsigned int)),     "Allocate memory for bone Ids");
    checkCudaErrorsMsg(cudaMalloc(&d_weightPtr,         m_numVerts * 4 * sizeof(float)),            "Allocate memory for bone weights");
    checkCudaErrorsMsg(cudaMalloc(&d_oneRingIdPtr,      oneRingIdFlat.size() * sizeof(int)),     "Allocate memory for one ring ids");
    checkCudaErrorsMsg(cudaMalloc(&d_oneRingVertPtr,    oneRingVertFlat.size() * sizeof(glm::vec3)), "Allocate memory for one ring verts");
    checkCudaErrorsMsg(cudaMalloc(&d_centroidWeightsPtr,   oneRingVertFlat.size() * sizeof(float)), "Allocate memory for one ring centroid weights");
    checkCudaErrorsMsg(cudaMalloc(&d_numNeighsPerVertPtr, (m_numVerts+1) * sizeof(int)),        "Allocate memory for num neighs per vertex");
    checkCudaErrorsMsg(cudaMalloc(&d_oneRingScatterAddrPtr, (m_numVerts+1) * sizeof(int)),      "Allocate memory for one ring scatter address");


    // copy memory over to cuda
    checkCudaErrors(cudaMemcpy((void*)d_origMeshVertsPtr, (void*)&_origMesh.m_meshVerts[0], m_numVerts * sizeof(glm::vec3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void*)d_origMeshNormsPtr, (void*)&_origMesh.m_meshNorms[0], m_numVerts * sizeof(glm::vec3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void*)d_transformPtr, (void*)&_transform[0][0][0], _transform.size() * sizeof(glm::mat4), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void*)d_boneIdPtr, (void*)boneIds, m_numVerts *4* sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void*)d_weightPtr, (void*)weights, m_numVerts *4* sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void*)d_oneRingIdPtr, (void*)&oneRingIdFlat[0], oneRingIdFlat.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void*)d_oneRingVertPtr, (void*)&oneRingVertFlat[0], oneRingVertFlat.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void*)d_numNeighsPerVertPtr, (void*)&numNeighsPerVertex[0], m_numVerts * sizeof(int), cudaMemcpyHostToDevice));
    kernels::GenerateScatterAddress(d_numNeighsPerVertPtr, (d_numNeighsPerVertPtr+m_numVerts+1), d_oneRingScatterAddrPtr);
//    kernels::GenerateOneRingCentroidWeights(d_origMeshVertsPtr, d_origMeshNormsPtr, m_numVerts, d_centroidWeightsPtr, d_oneRingIdPtr, d_oneRingVertPtr, d_numNeighsPerVertPtr, d_oneRingScatterAddrPtr);




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
    m_numFields = fieldFuncs.size();
    m_numCompOps = compOps.size();
    m_numCompFields = compFields.size();

    // allocate memory
    checkCudaErrors(cudaMalloc(&d_textureSpacePtr, m_numFields * sizeof(glm::mat4)));
    checkCudaErrors(cudaMalloc(&d_fieldsPtr, m_numFields * sizeof(cudaTextureObject_t)));
    checkCudaErrors(cudaMalloc(&d_compOpPtr, m_numCompOps * sizeof(cudaTextureObject_t)));
    checkCudaErrors(cudaMalloc(&d_compFieldPtr, m_numCompFields * sizeof(ComposedFieldCuda)));

    // upload data
    std::vector<glm::mat4> texSpaceTrans(m_numFields);
    for(int i=0; i<m_numFields; ++i)
    {
        texSpaceTrans[i] = fieldFuncs[i]->GetTextureSpaceTransform();
        auto fieldTex = fieldFuncs[i]->GetFieldFunc3DTexture();
        cudaMemcpy(d_fieldsPtr+i, &fieldTex, 1*sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    }
    for(int i=0; i<m_numCompOps; ++i)
    {
        auto compOpTex = compOps[i]->GetFieldFunc3DTexture();
        cudaMemcpy(d_compOpPtr+i, &compOpTex, 1*sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    }
    for(int i=0; i<m_numCompFields; ++i)
    {
        auto cf = compFields[i];
        cudaMemcpy(d_compFieldPtr+i, &cf, 1*sizeof(ComposedFieldCuda), cudaMemcpyHostToDevice);
    }
    checkCudaErrors(cudaMemcpy((void*)d_textureSpacePtr, &texSpaceTrans[0][0][0], texSpaceTrans.size() * sizeof(glm::mat4), cudaMemcpyHostToDevice));

    m_initFieldCudaMem = true;
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::DestroyMeshCudaMem()
{
    if(m_initMeshCudaMem)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(m_meshVBO_CUDA));
        checkCudaErrors(cudaGraphicsUnregisterResource(m_meshNBO_CUDA));
        checkCudaErrors(cudaFree(d_origMeshVertsPtr));
        checkCudaErrors(cudaFree(d_origMeshNormsPtr));
        checkCudaErrors(cudaFree(d_origVertIsoPtr));
        checkCudaErrors(cudaFree(d_newVertIsoPtr));
        checkCudaErrors(cudaFree(d_transformPtr));
        checkCudaErrors(cudaFree(d_boneIdPtr));
        checkCudaErrors(cudaFree(d_weightPtr));
        checkCudaErrors(cudaFree(d_oneRingIdPtr));
        checkCudaErrors(cudaFree(d_oneRingVertPtr));
        checkCudaErrors(cudaFree(d_centroidWeightsPtr));
        checkCudaErrors(cudaFree(d_numNeighsPerVertPtr));
        checkCudaErrors(cudaFree(d_oneRingScatterAddrPtr));

        m_initMeshCudaMem = false;
    }
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::DestroyFieldCudaMem()
{
    if(m_initFieldCudaMem)
    {
        checkCudaErrors(cudaFree(d_textureSpacePtr));
        checkCudaErrors(cudaFree(d_fieldsPtr));
        checkCudaErrors(cudaFree(d_compOpPtr));
        checkCudaErrors(cudaFree(d_compFieldPtr));

        m_initFieldCudaMem = false;
    }
}

//------------------------------------------------------------------------------------------------

GlobalFieldFunction &ImplicitSkinDeformer::GetGlocalFieldFunc()
{
    return m_globalFieldFunction;
}


//------------------------------------------------------------------------------------------------

glm::vec3 *ImplicitSkinDeformer::GetDeformedMeshVertsDevicePtr()
{
    if(!m_deformedMeshVertsMapped)
    {
        size_t numBytes;
        checkCudaErrors(cudaGraphicsMapResources(1, &m_meshVBO_CUDA, 0));
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_deformedMeshVertsPtr, &numBytes, m_meshVBO_CUDA));

        m_deformedMeshVertsMapped = true;
    }

    return d_deformedMeshVertsPtr;
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::ReleaseDeformedMeshVertsDevicePtr()
{
    if(m_deformedMeshVertsMapped)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &m_meshVBO_CUDA, 0));
        m_deformedMeshVertsMapped = false;
    }
}

//------------------------------------------------------------------------------------------------

glm::vec3 *ImplicitSkinDeformer::GetDeformedMeshNormsDevicePtr()
{
    if(!m_deformedMeshNormsMapped)
    {
        size_t numBytes;
        checkCudaErrors(cudaGraphicsMapResources(1, &m_meshNBO_CUDA, 0));
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_deformedMeshNormsPtr, &numBytes, m_meshNBO_CUDA));

        m_deformedMeshNormsMapped = true;
    }

    return d_deformedMeshNormsPtr;
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::ReleaseDeformedMeshNormsDevicePtr()
{
    if(m_deformedMeshNormsMapped)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &m_meshNBO_CUDA, 0));
        m_deformedMeshNormsMapped = false;
    }
}

//------------------------------------------------------------------------------------------------
