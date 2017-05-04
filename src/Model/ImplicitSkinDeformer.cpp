#include "Model/implicitskindeformer.h"
#include "ImplicitSkinGpuWrapper.h"
#include "helper_cuda.h"
#include <glm/gtx/string_cast.hpp>


//------------------------------------------------------------------------

ImplicitSkinDeformer::ImplicitSkinDeformer():
    m_initMeshCudaMem(false),
    m_initFieldCudaMem(false),
    m_deformedMeshVertsMapped(false),
    m_deformedMeshNormsMapped(false),
    m_sigma(0.35f),
    m_contactAngle(55.0f),
    m_numIterations(1)
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

void ImplicitSkinDeformer::InitialiseIsoValues()
{
    if(m_initFieldCudaMem && m_initMeshCudaMem & m_globalFieldFunction.IsGlobalFieldInit())
    {
        std::cout<<"init iso\n";
        // allocate and initialise temporary device memory
        glm::mat4 tmpTransform(1.0f);
        glm::mat4 * d_tmpTransform;
        checkCudaErrors(cudaMalloc(&d_tmpTransform, m_numTransforms * sizeof(glm::mat4)));
        for(int i=0;i<m_numTransforms;i++)
        {
            checkCudaErrors(cudaMemcpy(d_tmpTransform+i, &tmpTransform[0][0], sizeof(glm::mat4), cudaMemcpyHostToDevice));
        }


        // run kernel
        isgw::EvalGradGlobalField(d_origVertIsoPtr, d_vertIsoGradPtr, d_origMeshVertsPtr, m_numVerts, d_textureSpacePtr, d_tmpTransform, d_fieldsPtr, m_numFields, d_compOpPtr, d_thetaPtr, m_numCompOps, d_compFieldPtr, m_numCompFields);
        getLastCudaError("isgw::EvalGradGlobalField");


        // free temporary device memory
        checkCudaErrors(cudaFree(d_tmpTransform));
    }

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
                                                       const std::vector<std::pair<glm::vec3, glm::vec3> > &_boneEnds,
                                                       const int _numHrbfCentres)
{
    m_globalFieldFunction.Fit(_meshParts.size());
    float dim = FLT_MIN;
    dim = fabs(m_minBBox.x) > dim ? fabs(m_minBBox.x) : dim;
    dim = fabs(m_minBBox.y) > dim ? fabs(m_minBBox.y) : dim;
    dim = fabs(m_minBBox.z) > dim ? fabs(m_minBBox.z) : dim;
    dim = fabs(m_maxBBox.x) > dim ? fabs(m_maxBBox.x) : dim;
    dim = fabs(m_maxBBox.y) > dim ? fabs(m_maxBBox.y) : dim;
    dim = fabs(m_maxBBox.z) > dim ? fabs(m_maxBBox.z) : dim;
    dim *= 1.5f;
    int res = 64;

    // Generate individual field functions per mesh part
    auto threadFunc = [&, this](int startId, int endId){
        for(int mp=startId; mp<endId; mp++)
        {
            Mesh hrbfCentres;
            m_globalFieldFunction.GenerateHRBFCentres(_meshParts[mp], _boneEnds[mp], _numHrbfCentres, hrbfCentres);
            m_globalFieldFunction.GenerateFieldFuncs(hrbfCentres, _meshParts[mp], mp);
            m_globalFieldFunction.PrecomputeFieldFunc(mp, res, dim);

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


    InitFieldCudaMem();

    InitialiseIsoValues();
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::Deform()
{
    PerformLBWSkinning();
    PerformImplicitSkinning();


//    isgw::EvalGradGlobalField(d_newVertIsoPtr, d_vertIsoGradPtr, GetDeformedMeshVertsDevicePtr(), m_numVerts, d_textureSpacePtr, d_transformPtr, d_fieldsPtr, m_numFields, d_compOpPtr, d_thetaPtr, m_numCompOps, d_compFieldPtr, m_numCompFields);
//    getLastCudaError("isgw::EvalGradGlobalField");
//    ReleaseDeformedMeshVertsDevicePtr();

//    float h_iso[m_numVerts];
//    glm::vec3 h_grad[m_numVerts];
//    checkCudaErrors(cudaMemcpy(h_iso, d_newVertIsoPtr, m_numVerts*sizeof(float), cudaMemcpyDeviceToHost));
//    checkCudaErrors(cudaMemcpy(h_grad, d_vertIsoGradPtr, m_numVerts*sizeof(glm::vec3), cudaMemcpyDeviceToHost));
//    for(int i=0;i<m_numVerts;i++)
//    {
//        if(h_iso[i] > 0.2f && h_iso[i] < 0.8f)
//        {
//            std::cout<<h_iso[i]<<", "<<glm::to_string(h_grad[i])<<"\n";
//        }
//    }
//    std::cout<<"---------------------------------\n";

}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::PerformLBWSkinning()
{

    if(!m_initMeshCudaMem)
    {
        return;
    }


    isgw::LinearBlendWeightSkin(GetDeformedMeshVertsDevicePtr(),
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


    isgw::SimpleImplicitSkin(GetDeformedMeshVertsDevicePtr(), GetDeformedMeshNormsDevicePtr(), d_origVertIsoPtr, d_vertIsoGradPtr, m_numVerts,
                             d_textureSpacePtr, d_transformPtr, d_fieldsPtr, m_numFields,
                             d_compOpPtr, d_thetaPtr, m_numCompOps, d_compFieldPtr, m_numCompFields,
                             d_oneRingIdPtr, d_centroidWeightsPtr, d_oneRingScatterAddrPtr,
                             m_sigma, m_contactAngle, m_numIterations);



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
    if(m_globalFieldFunction.IsGlobalFieldInit())
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

void ImplicitSkinDeformer::SetSigma(float _sigma)
{
    m_sigma = _sigma;
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::SetContactAngle(float _contactAngle)
{
    m_contactAngle = _contactAngle;
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::SetIterations(int _iterations)
{
    m_numIterations = _iterations;
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::EvalGlobalField(std::vector<float> &_output, const std::vector<glm::vec3> &_samplePoints)
{
    _output.clear();
    if(!m_globalFieldFunction.IsGlobalFieldInit())
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

void ImplicitSkinDeformer::EvalGlobalFieldInCube(std::vector<float> &_output, const int res, const float dim)
{
    _output.clear();
    if(!m_globalFieldFunction.IsGlobalFieldInit())
    {
        return;
    }

    _output.resize(res *res *res);

    if(!m_initMeshCudaMem || !m_initFieldCudaMem)
    {
        EvalFieldInCubeCPU(_output, res, dim);
    }
    else
    {
        EvalFieldInCubeGPU(_output, res, dim);
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
    m_numFields = m_globalFieldFunction.GetFieldFuncs().size();

    // allocate device memory
    float *d_output;
    glm::vec3 *d_samplePoints;
    checkCudaErrors(cudaMalloc(&d_output, _samplePoints.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_samplePoints, _samplePoints.size() * sizeof(glm::vec3)));

    // upload data to device
    checkCudaErrors(cudaMemcpy((void*)d_samplePoints, &_samplePoints[0], _samplePoints.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice));

    // run kernel
    isgw::EvalGlobalField(d_output, d_samplePoints, _samplePoints.size(), d_textureSpacePtr, d_transformPtr, d_fieldsPtr, m_numFields, d_compOpPtr, d_thetaPtr, m_numCompOps, d_compFieldPtr, m_numCompFields);
    getLastCudaError("isgw::EvalGlobalField");

    // download data to host
    checkCudaErrors(cudaMemcpy(&_output[0], d_output, _samplePoints.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // free memory
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_samplePoints));
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::EvalFieldInCubeCPU(std::vector<float> &_output, const int res, const float dim)
{

}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::EvalFieldInCubeGPU(std::vector<float> &_output, const int res, const float dim)
{

}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::InitMeshCudaMem(const Mesh _origMesh,
                                           const GLuint _meshVBO,
                                           const GLuint _meshNBO,
                                           const std::vector<glm::mat4> &_transform)
{

    if(m_initMeshCudaMem) { return; }

    std::cout<<"init mesh cuda\n";

    m_numVerts = _origMesh.m_meshVerts.size();
    m_numTransforms = _transform.size();
    m_minBBox = _origMesh.m_minBBox;
    m_maxBBox = _origMesh.m_maxBBox;

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
    checkCudaErrorsMsg(cudaMalloc(&d_vertIsoGradPtr,    m_numVerts * sizeof(glm::vec3)),          "Allocate memory for new vert iso Grad values");
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
    isgw::GenerateScatterAddress(d_numNeighsPerVertPtr, (d_numNeighsPerVertPtr+m_numVerts+1), d_oneRingScatterAddrPtr);
    isgw::GenerateOneRingCentroidWeights(d_origMeshVertsPtr, d_origMeshNormsPtr, m_numVerts, d_centroidWeightsPtr, d_oneRingIdPtr, d_oneRingVertPtr, d_numNeighsPerVertPtr, d_oneRingScatterAddrPtr);


    m_initMeshCudaMem = true;
}

//------------------------------------------------------------------------------------------------

void ImplicitSkinDeformer::InitFieldCudaMem()
{
    if(!m_globalFieldFunction.IsGlobalFieldInit() || m_initFieldCudaMem) { return; }

    std::cout<<"init field cuda\n";

    auto fieldFuncs = m_globalFieldFunction.GetFieldFuncs();
    auto compOps = m_globalFieldFunction.GetCompOps();
    auto compFields = m_globalFieldFunction.GetCompFieldsCuda();

    // create device memory here
    m_numFields = fieldFuncs.size();
    m_numCompOps = compOps.size();
    m_numCompFields = compFields.size();

    // allocate memory
    checkCudaErrors(cudaMalloc(&d_textureSpacePtr, m_numFields * sizeof(glm::mat4)));
    checkCudaErrors(cudaMalloc(&d_fieldsPtr, m_numFields * sizeof(cudaTextureObject_t)));
    checkCudaErrors(cudaMalloc(&d_compOpPtr, m_numCompOps * sizeof(cudaTextureObject_t)));
    checkCudaErrors(cudaMalloc(&d_thetaPtr, m_numCompOps * sizeof(cudaTextureObject_t)));
    checkCudaErrors(cudaMalloc(&d_compFieldPtr, m_numCompFields * sizeof(ComposedFieldCuda)));

    // upload data
    std::vector<glm::mat4> texSpaceTrans(m_numFields);
    for(int i=0; i<m_numFields; ++i)
    {
        texSpaceTrans[i] = fieldFuncs[i]->GetTextureSpaceTransform();
        auto fieldTex = fieldFuncs[i]->GetFieldFuncCudaTextureObject();
        cudaMemcpy(d_fieldsPtr+i, &fieldTex, 1*sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    }
    for(int i=0; i<m_numCompOps; ++i)
    {
        auto compOpTex = compOps[i]->GetFieldFunc3DTexture();
        auto thetaTex = compOps[i]->GetThetaTexture();
        cudaMemcpy(d_compOpPtr+i, &compOpTex, 1*sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_thetaPtr+i, &thetaTex, 1*sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
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
        checkCudaErrors(cudaFree(d_vertIsoGradPtr));
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
        checkCudaErrors(cudaFree(d_thetaPtr));
        checkCudaErrors(cudaFree(d_compFieldPtr));

        m_initFieldCudaMem = false;
    }
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
