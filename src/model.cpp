#include "model.h"
#include "modelloader.h"
#include "MeshSampler/meshsampler.h"

#include <QOpenGLContext>

#include <sys/time.h>

#include <iostream>
#include <algorithm>
#include <stack>
#include <math.h>

#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/vector_angle.hpp>


GlobalFieldFunction Model::m_globalFieldFunction;

Model::Model()
{
    m_wireframe = false;
    m_initGL = false;
}

Model::~Model()
{

    for(int i=0; i<std::thread::hardware_concurrency()-1; i++)
    {
        if(m_threads[i].joinable())
        {
            m_threads[i].join();
        }
    }

    if(m_initGL)
    {
        DeleteVAOs();
    }
}

void Model::Load(const std::string &_file)
{
    ModelLoader::LoadModel(this, _file);
    //--------------------------------------------------

    GenerateMeshParts();
    GenerateFieldFunctions();
    GenerateGlobalFieldFunctions();
    GenerateMeshVertIsoValue();

    //--------------------------------------------------
    CreateShaders();
    CreateVAOs();
    UpdateVAOs();

    m_threads.resize(std::thread::hardware_concurrency()-1);
}


void Model::GenerateMeshParts()
{
    unsigned int numParts = m_rig.m_boneNameIdMapping.size();
    m_meshParts.resize(numParts);
    m_fieldFunctions.resize(numParts);


    for(unsigned int t=0; t<m_mesh.m_meshTris.size(); t++)
    {
        int v1 = m_mesh.m_meshTris[t].x;
        int v2 = m_mesh.m_meshTris[t].y;
        int v3 = m_mesh.m_meshTris[t].z;

        float weight[3] = {0.0f, 0.0f, 0.0f};
        int boneId[3] = {-1, -1, -1};
        for(int bw = 0; bw<4; bw++)
        {
            if(m_mesh.m_meshBoneWeights[v1].boneWeight[bw] > weight[0])
            {
                weight[0] = m_mesh.m_meshBoneWeights[v1].boneWeight[bw];
                boneId[0] = m_mesh.m_meshBoneWeights[v1].boneID[bw];
            }

            if(m_mesh.m_meshBoneWeights[v2].boneWeight[bw] > weight[1])
            {
                weight[1] = m_mesh.m_meshBoneWeights[v2].boneWeight[bw];
                boneId[1] = m_mesh.m_meshBoneWeights[v2].boneID[bw];
            }

            if(m_mesh.m_meshBoneWeights[v3].boneWeight[bw] > weight[2])
            {
                weight[2] = m_mesh.m_meshBoneWeights[v3].boneWeight[bw];
                boneId[2] = m_mesh.m_meshBoneWeights[v3].boneID[bw];
            }
        }

        for(unsigned int v=0; v<3; v++)
        {
            if(boneId[v] < 0 || boneId[v] >= (int)numParts)
            {
                continue;
            }
            if(v==1 && boneId[1] == boneId[0])
            {
                continue;
            }
            if((v==2) && (boneId[2] == boneId[1] || boneId[2] == boneId[0]))
            {
                continue;
            }

            if(weight[v] >0.2f)
                m_meshParts[boneId[v]].m_meshTris.push_back(glm::ivec3(v1, v2, v3));
        }

    }


    for(unsigned int i=0 ;i<m_meshParts.size(); i++)
    {
        m_meshParts[i].m_meshVerts = m_mesh.m_meshVerts;
        m_meshParts[i].m_meshNorms = m_mesh.m_meshNorms;
    }

}

void Model::GenerateFieldFunctions()
{
    m_fieldFunctions.resize(m_meshParts.size());


    unsigned int numHrbfFitPoints = 50;
    std::vector<HRBF::Vector> verts;
    std::vector<HRBF::Vector> norms;

    // iterate through mesh parts and initalise bone field functions
    for(unsigned int mp=0; mp<m_meshParts.size(); mp++)
    {
        verts.clear();
        norms.clear();


        // Generate HRBF centre by sampling mesh
        Mesh hrbfCentres = MeshSampler::BaryCoord::SampleMesh(m_meshParts[mp], numHrbfFitPoints);


        // Determine distance of closest point to bone
        glm::vec3 jointStart = m_rigMesh.m_meshVerts[mp*2];
        glm::vec3 jointEnd = m_rigMesh.m_meshVerts[(mp*2) + 1];
        glm::vec3 edge = jointEnd - jointStart;
        float minDist = FLT_MAX;
        for(auto &&tri : m_meshParts[mp].m_meshTris)
        {
            glm::vec3 v0 = m_meshParts[mp].m_meshVerts[tri.x];
            glm::vec3 v1 = m_meshParts[mp].m_meshVerts[tri.y];
            glm::vec3 v2 = m_meshParts[mp].m_meshVerts[tri.z];

            glm::vec3 e = v0 - jointStart;
            float t = glm::dot(e, edge);
            float dist = glm::distance(v0, jointStart + (t*edge));
            minDist = dist < minDist ? dist : minDist;

            e = v1 - jointStart;
            t = glm::dot(e, edge);
            dist = glm::distance(v1, jointStart + (t*edge));
            minDist = dist < minDist ? dist : minDist;

            e = v2 - jointStart;
            t = glm::dot(e, edge);
            dist = glm::distance(v2, jointStart + (t*edge));
            minDist = dist < minDist ? dist : minDist;
        }


        // Add these points to close holes of scalar field smoothly
        hrbfCentres.m_meshVerts.push_back(jointStart - (minDist * glm::normalize(edge)));
        hrbfCentres.m_meshNorms.push_back(-glm::normalize(edge));
        hrbfCentres.m_meshVerts.push_back(jointEnd + (minDist * glm::normalize(edge)));
        hrbfCentres.m_meshNorms.push_back(glm::normalize(edge));


        // Generate HRBF fit and thus scalar field/implicit function
        m_fieldFunctions[mp] = std::shared_ptr<FieldFunction>(new FieldFunction());
        m_fieldFunctions[mp]->Fit(hrbfCentres.m_meshVerts, hrbfCentres.m_meshNorms);


        // Find maximun range of scalar field
        float maxDist = FLT_MIN;
        for(auto &&tri : m_meshParts[mp].m_meshTris)
        {
            glm::vec3 v0 = m_meshParts[mp].m_meshVerts[tri.x];
            glm::vec3 v1 = m_meshParts[mp].m_meshVerts[tri.y];
            glm::vec3 v2 = m_meshParts[mp].m_meshVerts[tri.z];

            float f0 = m_fieldFunctions[mp]->EvalDist(v0);
            maxDist = f0 > maxDist ? f0 : maxDist;
            float f1 = m_fieldFunctions[mp]->EvalDist(v1);
            maxDist = f1 > maxDist ? f1 : maxDist;
            float f2 = m_fieldFunctions[mp]->EvalDist(v2);
            maxDist = f2 > maxDist ? f2 : maxDist;
        }


        // Set R in order to make field function compactly supported
        m_fieldFunctions[mp]->SetR(maxDist);
        m_fieldFunctions[mp]->PrecomputeField(64, 8.0f);
    }

}


void Model::GenerateGlobalFieldFunctions()
{
    // Time to build composition tree
    typedef std::shared_ptr<CompositionOp> CompositionOpPtr;

    // Initialise our various type of gradient based operators
    CompositionOpPtr contactOp = CompositionOpPtr(new CompositionOp());
    CompositionOpPtr bulgeOp = CompositionOpPtr(new CompositionOp());


    //TODO: Fit the operators so the dc(alpha) matches specific effect
//    contactOp->SetCompositionOp([](float f1, float f2, float d){
//        if(f1 > 0.7f || f2 > 0.7f) return f1 > f2 ? f1 : f2;

//        auto K = []()

//    });

    contactOp->SetTheta([](float _angleRadians){
        return _angleRadians <= M_PI ? (0.5f*(cosf(_angleRadians)+1.0f)) : 0.0f;
    });

//    bulgeOp->SetCompositionOp([](float f1, float f2, float d){

//    });

    bulgeOp->SetTheta([](float _angleRadians){
        return _angleRadians <= M_PI ? (0.5f*(cosf(2.0f*_angleRadians)+1.0f)) : 1.0f;
    });

    contactOp->Precompute(64);
    bulgeOp->Precompute(64);


    // ----------------TODO---------------------------------
    // Create and set correct unique composition operator
    //------------------------------------------------------
    // add composed fields to global field
    for(unsigned int mp=0; mp<m_fieldFunctions.size(); mp+=2)
    {
        int fieldId = 0;
        auto composedField = std::shared_ptr<ComposedField>(new ComposedField());
        composedField->SetCompositionOp(contactOp);

        if(m_meshParts[mp].m_meshTris.size() >0)
        {
//            auto fieldFunc1 = std::shared_ptr<FieldFunction>(&m_fieldFunctions[mp]);
            composedField->SetFieldFunc(m_fieldFunctions[mp], fieldId++);
        }

        if(m_fieldFunctions.size() > mp)
        {
            if(m_meshParts[mp+1].m_meshTris.size() > 0)
            {
//                auto fieldFunc2 = std::shared_ptr<FieldFunction>(&m_fieldFunctions[mp+1]);
                composedField->SetFieldFunc(m_fieldFunctions[mp+1], fieldId);
            }
        }

        m_globalFieldFunction.AddComposedField(composedField);
    }
}

//---------------------------------------------------------------------------------

void Model::GenerateMeshVertIsoValue()
{
    m_meshVertIsoValues.clear();
    for(auto &v : m_mesh.m_meshVerts)
    {
        m_meshVertIsoValues.push_back(m_globalFieldFunction.Eval(v));
    }
}

void Model::PerformLBWSkinning()
{

}

void Model::PerformVertexProjection()
{
    std::vector<glm::vec3> newVert(m_mesh.m_meshVerts.size());
    std::vector<glm::vec3> previousGrad(m_mesh.m_meshVerts.size());
    std::vector<float> gradAngle(m_mesh.m_meshVerts.size());
    float sigma = 0.35f;
    float contactAngle = 55.0f;

    // TODO
    // Replace m_mesh.m_meshVerts[i] with LBW Skinned Tranformed Vert

    for(int i =0; i<m_mesh.m_meshVerts.size(); i++)
    {
        glm::vec3 grad = m_globalFieldFunction.Grad(m_mesh.m_meshVerts[i]);
        float angle = gradAngle[i] = glm::angle(grad, previousGrad[i]);
        if(angle < contactAngle)
        {
            newVert[i] = m_mesh.m_meshVerts[i] + ( sigma * (m_globalFieldFunction.Eval(m_mesh.m_meshVerts[i]) - m_meshVertIsoValues[i]) * (grad / glm::length2(grad)));
            previousGrad[i] = grad;
        }
    }
}

void Model::PerformTangentialRelaxation()
{
    std::vector<glm::vec3> newVert(m_mesh.m_meshVerts.size());

    for(int i =0; i<m_mesh.m_meshVerts.size(); i++)
    {
        glm::vec3 origVert = m_mesh.m_meshVerts[i];
        float newIsoValue = m_globalFieldFunction.Eval(m_mesh.m_meshVerts[i]);
        float mu = std::max(0.0f, 1.0f - (float)pow(fabs(newIsoValue - m_meshVertIsoValues[i]) - 1.0f, 4.0f));

        glm::vec3 sumWeightedCentroid(0.0f);
        int j=0;
        for(auto &n : m_meshVertOneRingNeighbour[i])
        {
            glm::vec3 neighVert = m_mesh.m_meshVerts[n];
            glm::vec3 projNeighVert = neighVert;
            float barycentricCoord = m_meshVertCentroidWeights[i][j];
            sumWeightedCentroid += barycentricCoord * projNeighVert;

            j++;
        }

        newVert[i] = ((1.0f - mu) * origVert) + (mu * sumWeightedCentroid);
    }
}

void Model::PerformLaplacianSmoothing()
{
}

//---------------------------------------------------------------------------------

void Model::UpdateImplicitSurface(int xRes,
                                  int yRes,
                                  int zRes,
                                  float dim,
                                  float xScale,
                                  float yScale,
                                  float zScale)
{

//    static double time = 0.0;
//    static double t1 = 0.0;
//    static double t2 = 0.0;
//    struct timeval tim;

//    gettimeofday(&tim, NULL);
//    t1=tim.tv_sec+(tim.tv_usec/1000000.0);


    for(unsigned int mp=0; mp<m_fieldFunctions.size(); mp++)
    {
        if(m_rig.m_boneTransforms.size() <= mp)
        {
            continue;
        }

        m_fieldFunctions[mp]->SetTransform(glm::inverse(m_rig.m_boneTransforms[mp]));
    }

    float *volumeData = new float[xRes*yRes*zRes];

    int numThreads = std::thread::hardware_concurrency();
    int chunkSize = zRes / numThreads;
    int startChunk = 0;

    for(int i=0; i<numThreads-1; i++)
    {
        m_threads[i] = std::thread([startChunk, &chunkSize, &dim, &xRes, &yRes, zRes, &m_globalFieldFunction, &volumeData](){
            for(int z=startChunk;z<startChunk+chunkSize;z++)
            {
                for(int y=0;y<yRes;y++)
                {
                    for(int x=0;x<xRes;x++)
                    {
                        glm::vec3 point(dim*((((float)x/zRes)*2.0f)-1.0f),
                                        dim*((((float)y/yRes)*2.0f)-1.0f),
                                        dim*((((float)z/xRes)*2.0f)-1.0f));

                        float d = m_globalFieldFunction.Eval(point);

                        if(!std::isnan(d))
                        {
                            volumeData[z*xRes*yRes + y*xRes + x] = d;
                        }
                        else
                        {
                            volumeData[z*xRes*yRes + y*xRes + x] = 0.0f;
                        }
                    }
                }
            }
        });

        startChunk += chunkSize;
    }


    for(int z=startChunk;z<zRes;z++)
    {
        for(int y=0;y<yRes;y++)
        {
            for(int x=0;x<xRes;x++)
            {
                glm::vec3 point(dim*((((float)x/zRes)*2.0f)-1.0f),
                                dim*((((float)y/yRes)*2.0f)-1.0f),
                                dim*((((float)z/xRes)*2.0f)-1.0f));

                float d = m_globalFieldFunction.Eval(point);

                if(!std::isnan(d))
                {
                    volumeData[z*xRes*yRes + y*xRes + x] = d;
                }
                else
                {
                    volumeData[z*xRes*yRes + y*xRes + x] = 0.0f;
                }
            }
        }
    }

    for(int i=0; i<numThreads-1; i++)
    {
        if(m_threads[i].joinable())
        {
            m_threads[i].join();
        }
    }


//    gettimeofday(&tim, NULL);
//    t2=tim.tv_sec+(tim.tv_usec/1000000.0);
//    time += 10*(t2-t1);
//    std::cout<<"evaluate global field: "<<1.0/(t2-t1)<<"\n";

    // Polygonize scalar field using maching cube
    m_polygonizer.Polygonize(m_meshIsoSurface.m_meshVerts, m_meshIsoSurface.m_meshNorms, volumeData, 0.5f, xRes, yRes, zRes, xScale, yScale, zScale);


    //clean up
    if(volumeData != nullptr)
    {
        delete volumeData;
    }
}


void Model::DrawMesh()
{



    static float angle = 0.0f;
    angle+=0.1f;
    if(!m_initGL)
    {
        CreateVAOs();
        UpdateVAOs();
    }
    else
    {
        //-------------------------------------------------------------------------------------
        // Draw skinned mesh

        m_shaderProg[SKINNED]->bind();
        glUniformMatrix4fv(m_projMatrixLoc[SKINNED], 1, false, &m_projMat[0][0]);
        glUniformMatrix4fv(m_mvMatrixLoc[SKINNED], 1, false, &(m_viewMat*m_modelMat)[0][0]);
        glm::mat3 normalMatrix =  glm::inverse(glm::mat3(m_modelMat));
        glUniformMatrix3fv(m_normalMatrixLoc[SKINNED], 1, true, &normalMatrix[0][0]);
        glUniform3fv(m_colourLoc[SKINNED], 1, &m_mesh.m_colour[0]);
        if(!m_wireframe)
        {
        m_meshVAO[SKINNED].bind();
        glPolygonMode(GL_FRONT_AND_BACK, m_wireframe?GL_LINE:GL_FILL);
        glDrawElements(GL_TRIANGLES, 3*m_mesh.m_meshTris.size(), GL_UNSIGNED_INT, &m_mesh.m_meshTris[0]);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        m_meshVAO[SKINNED].release();
        m_shaderProg[SKINNED]->release();
        }


        //-------------------------------------------------------------------------------------
        // Draw implicit mesh
        m_shaderProg[ISO_SURFACE]->bind();
        glUniformMatrix4fv(m_projMatrixLoc[ISO_SURFACE], 1, false, &m_projMat[0][0]);
        glUniformMatrix4fv(m_mvMatrixLoc[ISO_SURFACE], 1, false, &(m_modelMat*m_viewMat)[0][0]);
        normalMatrix =  glm::mat3(1.0f);//glm::inverse(glm::mat3(m_modelMat));
        glUniformMatrix3fv(m_normalMatrixLoc[ISO_SURFACE], 1, true, &normalMatrix[0][0]);

        // Get Scalar field for each mesh part and polygonize
        int xRes = 32;
        int yRes = 32;
        int zRes = 32;
        float dim = 800.0f; // dimension of sample range e.g. dim x dim x dim
        float xScale = 1.0f* dim;
        float yScale = 1.0f* dim;
        float zScale = 1.0f* dim;
        UpdateImplicitSurface(xRes, yRes, zRes, dim, xScale, yScale, zScale);

        // Global IsoSurface
        // upload new verts
        glUniform3fv(m_colourLoc[ISO_SURFACE], 1, &m_meshIsoSurface.m_colour[0]);// Setup our vertex buffer object.
        m_meshIsoVBO->bind();
        m_meshIsoVBO->allocate(&m_meshIsoSurface.m_meshVerts[0], m_meshIsoSurface.m_meshVerts.size() * sizeof(glm::vec3));
        glEnableVertexAttribArray(m_vertAttrLoc[ISO_SURFACE]);
        glVertexAttribPointer(m_vertAttrLoc[ISO_SURFACE], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshIsoVBO->release();


        // upload new normals
        m_meshIsoNBO->bind();
        m_meshIsoNBO->allocate(&m_meshIsoSurface.m_meshNorms[0], m_meshIsoSurface.m_meshNorms.size() * sizeof(glm::vec3));
        glEnableVertexAttribArray(m_normAttrLoc[ISO_SURFACE]);
        glVertexAttribPointer(m_normAttrLoc[ISO_SURFACE], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshIsoNBO->release();


        // Draw marching cube of isosurface
        m_meshIsoVAO->bind();
        glPolygonMode(GL_FRONT_AND_BACK, m_wireframe?GL_FILL:GL_FILL);
        glDrawArrays(GL_TRIANGLES, 0, m_meshIsoSurface.m_meshVerts.size());
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        m_meshIsoVAO->release();

        m_shaderProg[ISO_SURFACE]->release();
    }

}

void Model::DrawRig()
{
    if(!m_initGL)
    {
        CreateVAOs();
        UpdateVAOs();
    }
    else
    {
        m_shaderProg[RIG]->bind();
        glUniformMatrix4fv(m_projMatrixLoc[RIG], 1, false, &m_projMat[0][0]);
        glUniformMatrix4fv(m_mvMatrixLoc[RIG], 1, false, &(m_modelMat*m_viewMat)[0][0]);
        glm::mat3 normalMatrix =  glm::inverse(glm::mat3(m_modelMat));
        glUniformMatrix3fv(m_normalMatrixLoc[RIG], 1, true, &normalMatrix[0][0]);

        m_meshVAO[RIG].bind();
        glPolygonMode(GL_FRONT_AND_BACK, m_wireframe?GL_LINE:GL_FILL);
        glDrawArrays(GL_LINES, 0, m_rigMesh.m_meshVerts.size());
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        m_meshVAO[RIG].release();

        m_shaderProg[RIG]->release();
    }
}

void Model::Animate(const float _animationTime)
{
    m_rig.Animate(_animationTime);


    m_shaderProg[SKINNED]->bind();
    UploadBonesToShader(SKINNED);
    UploadBoneColoursToShader(SKINNED);
    m_shaderProg[SKINNED]->release();

    m_shaderProg[RIG]->bind();
    UploadBonesToShader(RIG);
    m_shaderProg[RIG]->release();
}

void Model::ToggleWireframe()
{
    m_wireframe = !m_wireframe;
}


void Model::CreateShaders()
{
    // SKINNING Shader
    m_shaderProg[SKINNED] = new QOpenGLShaderProgram();
    m_shaderProg[SKINNED]->addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/skinningVert.glsl");
    m_shaderProg[SKINNED]->addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/skinningFrag.glsl");
    m_shaderProg[SKINNED]->bindAttributeLocation("vertex", 0);
    m_shaderProg[SKINNED]->bindAttributeLocation("normal", 1);
    if(!m_shaderProg[SKINNED]->link())
    {
        std::cout<<m_shaderProg[SKINNED]->log().toStdString()<<"\n";
    }

    m_shaderProg[SKINNED]->bind();
    m_projMatrixLoc[SKINNED] = m_shaderProg[SKINNED]->uniformLocation("projMatrix");
    m_mvMatrixLoc[SKINNED] = m_shaderProg[SKINNED]->uniformLocation("mvMatrix");
    m_normalMatrixLoc[SKINNED] = m_shaderProg[SKINNED]->uniformLocation("normalMatrix");
    m_lightPosLoc[SKINNED] = m_shaderProg[SKINNED]->uniformLocation("lightPos");

    // Light position is fixed.
    m_lightPos = glm::vec3(0, 0, 70);
    glUniform3fv(m_lightPosLoc[SKINNED], 1, &m_lightPos[0]);
    m_shaderProg[SKINNED]->release();


    //------------------------------------------------------------------------------------------------
    // RIG Shader
    m_shaderProg[RIG] = new QOpenGLShaderProgram();
    m_shaderProg[RIG]->addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/rigVert.glsl");
    m_shaderProg[RIG]->addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/rigFrag.glsl");
    m_shaderProg[RIG]->addShaderFromSourceFile(QOpenGLShader::Geometry, "../shader/rigGeo.glsl");
    m_shaderProg[RIG]->bindAttributeLocation("vertex", 0);
    if(!m_shaderProg[RIG]->link())
    {
        std::cout<<m_shaderProg[RIG]->log().toStdString()<<"\n";
    }

    m_shaderProg[RIG]->bind();
    m_projMatrixLoc[RIG] = m_shaderProg[RIG]->uniformLocation("projMatrix");
    m_mvMatrixLoc[RIG] = m_shaderProg[RIG]->uniformLocation("mvMatrix");
    m_normalMatrixLoc[RIG] = m_shaderProg[RIG]->uniformLocation("normalMatrix");
    m_shaderProg[RIG]->release();



    //------------------------------------------------------------------------------------------------
    // ISO SURFACE Shader
    m_shaderProg[ISO_SURFACE] = new QOpenGLShaderProgram();
    m_shaderProg[ISO_SURFACE]->addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/vert.glsl");
    m_shaderProg[ISO_SURFACE]->addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/frag.glsl");
    m_shaderProg[ISO_SURFACE]->bindAttributeLocation("vertex", 0);
    if(!m_shaderProg[ISO_SURFACE]->link())
    {
        std::cout<<m_shaderProg[ISO_SURFACE]->log().toStdString()<<"\n";
    }

    m_shaderProg[ISO_SURFACE]->bind();
    m_projMatrixLoc[ISO_SURFACE] = m_shaderProg[ISO_SURFACE]->uniformLocation("projMatrix");
    m_mvMatrixLoc[ISO_SURFACE] = m_shaderProg[ISO_SURFACE]->uniformLocation("mvMatrix");
    m_normalMatrixLoc[ISO_SURFACE] = m_shaderProg[ISO_SURFACE]->uniformLocation("normalMatrix");
    m_lightPosLoc[SKINNED] = m_shaderProg[ISO_SURFACE]->uniformLocation("lightPos");
    // Light position is fixed.
    m_lightPos = glm::vec3(0, 0, 70);
    glUniform3fv(m_lightPosLoc[ISO_SURFACE], 1, &m_lightPos[0]);
    m_shaderProg[ISO_SURFACE]->release();


    m_initGL = true;
}

void Model::DeleteShaders()
{
    for(unsigned int i=0; i<NUMRENDERTYPES; i++)
    {
        if(m_shaderProg != nullptr)
        {
            m_shaderProg[i]->release();
            m_shaderProg[i]->destroyed();
            delete m_shaderProg[i];
        }
    }
}

void Model::CreateVAOs()
{
    //--------------------------------------------------------------------------------------
    // skinned mesh
    if(m_shaderProg[SKINNED]->bind())
    {
        // Get shader locations
        m_mesh.m_colour = glm::vec3(0.4f,0.4f,0.4f);
        m_colourLoc[SKINNED] = m_shaderProg[SKINNED]->uniformLocation("uColour");
        glUniform3fv(m_colourLoc[SKINNED], 1, &m_mesh.m_colour[0]);
        m_vertAttrLoc[SKINNED] = m_shaderProg[SKINNED]->attributeLocation("vertex");
        m_normAttrLoc[SKINNED] = m_shaderProg[SKINNED]->attributeLocation("normal");
        m_boneIDAttrLoc[SKINNED] = m_shaderProg[SKINNED]->attributeLocation("BoneIDs");
        m_boneWeightAttrLoc[SKINNED] = m_shaderProg[SKINNED]->attributeLocation("Weights");
        m_boneUniformLoc[SKINNED] = m_shaderProg[SKINNED]->uniformLocation("Bones");
        m_colourAttrLoc[SKINNED] = m_shaderProg[SKINNED]->uniformLocation("BoneColours");

        // Set up VAO
        m_meshVAO[SKINNED].create();
        m_meshVAO[SKINNED].bind();

        // Set up element array
        m_meshIBO[SKINNED].create();
        m_meshIBO[SKINNED].bind();
        m_meshIBO[SKINNED].release();


        // Setup our vertex buffer object.
        m_meshVBO[SKINNED].create();
        m_meshVBO[SKINNED].bind();
        glEnableVertexAttribArray(m_vertAttrLoc[SKINNED]);
        glVertexAttribPointer(m_vertAttrLoc[SKINNED], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshVBO[SKINNED].release();


        // Setup our normals buffer object.
        m_meshNBO[SKINNED].create();
        m_meshNBO[SKINNED].bind();
        glEnableVertexAttribArray(m_normAttrLoc[SKINNED]);
        glVertexAttribPointer(m_normAttrLoc[SKINNED], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshNBO[SKINNED].release();


        // Set up vertex bone weighting buffer object
        m_meshBWBO[SKINNED].create();
        m_meshBWBO[SKINNED].bind();
        glEnableVertexAttribArray(m_boneIDAttrLoc[SKINNED]);
        glVertexAttribIPointer(m_boneIDAttrLoc[SKINNED], MaxNumBlendWeightsPerVertex, GL_UNSIGNED_INT, sizeof(VertexBoneData), (const GLvoid*)0);
        glEnableVertexAttribArray(m_boneWeightAttrLoc[SKINNED]);
        glVertexAttribPointer(m_boneWeightAttrLoc[SKINNED], MaxNumBlendWeightsPerVertex, GL_FLOAT, GL_FALSE, sizeof(VertexBoneData), (const GLvoid*)(MaxNumBlendWeightsPerVertex*sizeof(unsigned int)));
        m_meshBWBO[SKINNED].release();

        m_meshVAO[SKINNED].release();

        m_shaderProg[SKINNED]->release();

    }


    //--------------------------------------------------------------------------------------
    // Rigged mesh
    if(m_shaderProg[RIG]->bind())
    {
        // Get shader locations
        m_mesh.m_colour = glm::vec3(0.4f,0.4f,0.4f);
        m_vertAttrLoc[RIG] = m_shaderProg[RIG]->attributeLocation("vertex");
        m_boneIDAttrLoc[RIG] = m_shaderProg[RIG]->attributeLocation("BoneIDs");
        m_boneWeightAttrLoc[RIG] = m_shaderProg[RIG]->attributeLocation("Weights");
        m_boneUniformLoc[RIG] = m_shaderProg[RIG]->uniformLocation("Bones");
        m_colourAttrLoc[RIG] = m_shaderProg[RIG]->attributeLocation("colour");

        m_meshVAO[RIG].create();
        m_meshVAO[RIG].bind();

        // Setup our vertex buffer object.
        m_meshVBO[RIG].create();
        m_meshVBO[RIG].bind();
        glEnableVertexAttribArray(m_vertAttrLoc[RIG]);
        glVertexAttribPointer(m_vertAttrLoc[RIG], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshVBO[RIG].release();

        // Set up our Rig joint colour buffer object
        m_meshCBO[RIG].create();
        m_meshCBO[RIG].bind();
        glEnableVertexAttribArray(m_colourAttrLoc[RIG]);
        glVertexAttribPointer(m_colourAttrLoc[RIG], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshCBO[RIG].release();

        // Set up vertex bone weighting buffer object
        m_meshBWBO[RIG].create();
        m_meshBWBO[RIG].bind();
        glEnableVertexAttribArray(m_boneIDAttrLoc[RIG]);
        glVertexAttribIPointer(m_boneIDAttrLoc[RIG], MaxNumBlendWeightsPerVertex, GL_UNSIGNED_INT, sizeof(VertexBoneData), (const GLvoid*)0);
        glEnableVertexAttribArray(m_boneWeightAttrLoc[RIG]);
        glVertexAttribPointer(m_boneWeightAttrLoc[RIG], MaxNumBlendWeightsPerVertex, GL_FLOAT, GL_FALSE, sizeof(VertexBoneData), (const GLvoid*) (MaxNumBlendWeightsPerVertex*sizeof(unsigned int)));
        m_meshBWBO[RIG].release();

        m_meshVAO[RIG].release();

        m_shaderProg[RIG]->release();
    }


    //------------------------------------------------------------------------------------
    // Iso surface Global
    if(m_shaderProg[ISO_SURFACE]->bind())
    {
        // Get shader locations
        m_mesh.m_colour = glm::vec3(0.4f,0.4f,0.4f);
        m_colourLoc[ISO_SURFACE] = m_shaderProg[ISO_SURFACE]->uniformLocation("uColour");
        glUniform3fv(m_colourLoc[ISO_SURFACE], 1, &m_mesh.m_colour[0]);
        m_vertAttrLoc[ISO_SURFACE] = m_shaderProg[ISO_SURFACE]->attributeLocation("vertex");
        m_normAttrLoc[ISO_SURFACE] = m_shaderProg[ISO_SURFACE]->attributeLocation("normal");

        // Set up VAO
        m_meshVAO[ISO_SURFACE].create();
        m_meshVAO[ISO_SURFACE].bind();


        // Setup our vertex buffer object.
        m_meshVBO[ISO_SURFACE].create();
        m_meshVBO[ISO_SURFACE].bind();
        glEnableVertexAttribArray(m_vertAttrLoc[ISO_SURFACE]);
        glVertexAttribPointer(m_vertAttrLoc[ISO_SURFACE], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshVBO[ISO_SURFACE].release();


        // Setup our normals buffer object.
        m_meshNBO[ISO_SURFACE].create();
        m_meshNBO[ISO_SURFACE].bind();
        glEnableVertexAttribArray(m_normAttrLoc[ISO_SURFACE]);
        glVertexAttribPointer(m_normAttrLoc[ISO_SURFACE], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshNBO[ISO_SURFACE].release();


        m_meshVAO[ISO_SURFACE].release();

        m_shaderProg[ISO_SURFACE]->release();

    }


    //------------------------------------------------------------------------------------
    // Iso surface MeshParts
    if(m_shaderProg[ISO_SURFACE]->bind())
    {
        // Global IsoSurface stuff
        m_meshIsoVAO = std::shared_ptr<QOpenGLVertexArrayObject>(new QOpenGLVertexArrayObject());
        m_meshIsoVBO = std::shared_ptr<QOpenGLBuffer>(new QOpenGLBuffer());
        m_meshIsoNBO = std::shared_ptr<QOpenGLBuffer>(new QOpenGLBuffer());


        // Get shader locations
        m_mesh.m_colour = glm::vec3(0.4f,0.4f,0.4f);
        m_colourLoc[ISO_SURFACE] = m_shaderProg[ISO_SURFACE]->uniformLocation("uColour");
        glUniform3fv(m_colourLoc[ISO_SURFACE], 1, &m_mesh.m_colour[0]);
        m_vertAttrLoc[ISO_SURFACE] = m_shaderProg[ISO_SURFACE]->attributeLocation("vertex");
        m_normAttrLoc[ISO_SURFACE] = m_shaderProg[ISO_SURFACE]->attributeLocation("normal");

        // Set up VAO
        m_meshIsoVAO->create();
        m_meshIsoVAO->bind();


        // Setup our vertex buffer object.
        m_meshIsoVBO->create();
        m_meshIsoVBO->bind();
        glEnableVertexAttribArray(m_vertAttrLoc[ISO_SURFACE]);
        glVertexAttribPointer(m_vertAttrLoc[ISO_SURFACE], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshIsoVBO->release();


        // Setup our normals buffer object.
        m_meshIsoNBO->create();
        m_meshIsoNBO->bind();
        glEnableVertexAttribArray(m_normAttrLoc[ISO_SURFACE]);
        glVertexAttribPointer(m_normAttrLoc[ISO_SURFACE], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshIsoNBO->release();


        m_meshIsoVAO->release();

        m_shaderProg[ISO_SURFACE]->release();

    }


    m_initGL = true;
}

void Model::DeleteVAOs()
{

    for(unsigned int i=0; i<NUMRENDERTYPES; i++)
    {
        if(m_meshVBO[i].isCreated())
        {
            m_meshVBO[i].destroy();
        }

        if(m_meshNBO[i].isCreated())
        {
            m_meshNBO[i].destroy();
        }

        if(m_meshIBO[i].isCreated())
        {
            m_meshIBO[i].destroy();
        }

        if(m_meshBWBO[i].isCreated())
        {
            m_meshBWBO[i].destroy();
        }

        if(m_meshVAO[i].isCreated())
        {
            m_meshVAO[i].destroy();
        }
    }
}

void Model::UpdateVAOs()
{
    // Skinned mesh
    if(m_shaderProg[SKINNED]->bind())
    {
        // Get shader locations
        glUniform3fv(m_colourLoc[SKINNED], 1, &m_mesh.m_colour[0]);

        // Set up VAO
        m_meshVAO[SKINNED].bind();

        // Set up element array
        m_meshIBO[SKINNED].bind();
        m_meshIBO[SKINNED].allocate(&m_mesh.m_meshTris[0], m_mesh.m_meshTris.size() * sizeof(int));
        m_meshIBO[SKINNED].release();


        // Setup our vertex buffer object.
        m_meshVBO[SKINNED].bind();
        m_meshVBO[SKINNED].allocate(&m_mesh.m_meshVerts[0], m_mesh.m_meshVerts.size() * sizeof(glm::vec3));
        glEnableVertexAttribArray(m_vertAttrLoc[SKINNED]);
        glVertexAttribPointer(m_vertAttrLoc[SKINNED], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshVBO[SKINNED].release();


        // Setup our normals buffer object.
        m_meshNBO[SKINNED].bind();
        m_meshNBO[SKINNED].allocate(&m_mesh.m_meshNorms[0], m_mesh.m_meshNorms.size() * sizeof(glm::vec3));
        glEnableVertexAttribArray(m_normAttrLoc[SKINNED]);
        glVertexAttribPointer(m_normAttrLoc[SKINNED], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshNBO[SKINNED].release();


        // Set up vertex bone weighting buffer object
        m_meshBWBO[SKINNED].bind();
        m_meshBWBO[SKINNED].allocate(&m_mesh.m_meshBoneWeights[0], m_mesh.m_meshBoneWeights.size() * sizeof(VertexBoneData));
        glEnableVertexAttribArray(m_boneIDAttrLoc[SKINNED]);
        glVertexAttribIPointer(m_boneIDAttrLoc[SKINNED], MaxNumBlendWeightsPerVertex, GL_UNSIGNED_INT, sizeof(VertexBoneData), (const GLvoid*)0);
        glEnableVertexAttribArray(m_boneWeightAttrLoc[SKINNED]);
        glVertexAttribPointer(m_boneWeightAttrLoc[SKINNED], MaxNumBlendWeightsPerVertex, GL_FLOAT, GL_FALSE, sizeof(VertexBoneData), (const GLvoid*)(MaxNumBlendWeightsPerVertex*sizeof(unsigned int)));
        m_meshBWBO[SKINNED].release();

        m_meshVAO[SKINNED].release();
        m_shaderProg[SKINNED]->release();
    }



    //--------------------------------------------------------------------------------------
    // Rigged mesh
    if(m_shaderProg[RIG]->bind())
    {
        m_meshVAO[RIG].bind();

        // Setup our vertex buffer object.
        m_meshVBO[RIG].bind();
        m_meshVBO[RIG].allocate(&m_rigMesh.m_meshVerts[0], m_rigMesh.m_meshVerts.size() * sizeof(glm::vec3));
        glEnableVertexAttribArray(m_vertAttrLoc[RIG]);
        glVertexAttribPointer(m_vertAttrLoc[RIG], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshVBO[RIG].release();

        // Set up our Rig joint colour buffer object
        m_meshCBO[RIG].bind();
        m_meshCBO[RIG].allocate(&m_rigMesh.m_meshVertColours[0], m_rigMesh.m_meshVertColours.size() * sizeof(glm::vec3));
        glEnableVertexAttribArray(m_colourAttrLoc[RIG]);
        glVertexAttribPointer(m_colourAttrLoc[RIG], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshCBO[RIG].release();

        // Set up vertex bone weighting buffer object
        m_meshBWBO[RIG].bind();
        m_meshBWBO[RIG].allocate(&m_rigMesh.m_meshBoneWeights[0], m_rigMesh.m_meshBoneWeights.size() * sizeof(VertexBoneData));
        glEnableVertexAttribArray(m_boneIDAttrLoc[RIG]);
        glVertexAttribIPointer(m_boneIDAttrLoc[RIG], MaxNumBlendWeightsPerVertex, GL_UNSIGNED_INT, sizeof(VertexBoneData), (const GLvoid*)0);
        glEnableVertexAttribArray(m_boneWeightAttrLoc[RIG]);
        glVertexAttribPointer(m_boneWeightAttrLoc[RIG], MaxNumBlendWeightsPerVertex, GL_FLOAT, GL_FALSE, sizeof(VertexBoneData), (const GLvoid*) (MaxNumBlendWeightsPerVertex*sizeof(unsigned int)));
        m_meshBWBO[RIG].release();

        m_meshVAO[RIG].release();

        m_shaderProg[RIG]->release();
    }



    //--------------------------------------------------------------------------------------
    // ISO SURFACE Global mesh
    if(m_shaderProg[ISO_SURFACE]->bind())
    {
        m_meshVAO[ISO_SURFACE].bind();

        // Setup our vertex buffer object.
        m_meshVBO[ISO_SURFACE].bind();
        m_meshVBO[ISO_SURFACE].allocate(&m_meshIsoSurface.m_meshVerts[0], m_meshIsoSurface.m_meshVerts.size() * sizeof(glm::vec3));
        glEnableVertexAttribArray(m_vertAttrLoc[ISO_SURFACE]);
        glVertexAttribPointer(m_vertAttrLoc[ISO_SURFACE], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshVBO[ISO_SURFACE].release();


        // Setup our normals buffer object.
        m_meshNBO[ISO_SURFACE].bind();
        m_meshNBO[ISO_SURFACE].allocate(&m_meshIsoSurface.m_meshNorms[0], m_meshIsoSurface.m_meshNorms.size() * sizeof(glm::vec3));
        glEnableVertexAttribArray(m_normAttrLoc[ISO_SURFACE]);
        glVertexAttribPointer(m_normAttrLoc[ISO_SURFACE], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshNBO[ISO_SURFACE].release();


        m_meshVAO[ISO_SURFACE].release();

        m_shaderProg[ISO_SURFACE]->release();
    }


    //--------------------------------------------------------------------------------------
    // ISO SURFACE MeshParts
    if(m_shaderProg[ISO_SURFACE]->bind())
    {
        // Global IsoSurface
        m_meshIsoVAO->bind();

        // Setup our vertex buffer object.
        m_meshIsoVBO->bind();
        m_meshIsoVBO->allocate(&m_meshIsoSurface.m_meshVerts[0], m_meshIsoSurface.m_meshVerts.size() * sizeof(glm::vec3));
        glEnableVertexAttribArray(m_vertAttrLoc[ISO_SURFACE]);
        glVertexAttribPointer(m_vertAttrLoc[ISO_SURFACE], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshIsoVBO->release();


        // Setup our normals buffer object.
        m_meshIsoNBO->bind();
        m_meshIsoNBO->allocate(&m_meshIsoSurface.m_meshNorms[0], m_meshIsoSurface.m_meshNorms.size() * sizeof(glm::vec3));
        glEnableVertexAttribArray(m_normAttrLoc[ISO_SURFACE]);
        glVertexAttribPointer(m_normAttrLoc[ISO_SURFACE], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshIsoNBO->release();


        m_meshIsoVAO->release();

        m_shaderProg[ISO_SURFACE]->release();
    }
}



void Model::SetLightPos(const glm::vec3 &_lightPos)
{
    m_lightPos = _lightPos;
}

void Model::SetModelMatrix(const glm::mat4 &_modelMat)
{
    m_modelMat = _modelMat;
}

void Model::SetNormalMatrix(const glm::mat3 &_normMat)
{
    m_normMat = _normMat;
}

void Model::SetViewMatrix(const glm::mat4 &_viewMat)
{
    m_viewMat = _viewMat;
}

void Model::SetProjectionMatrix(const glm::mat4 &_projMat)
{
    m_projMat = _projMat;
}

void Model::UploadBoneColoursToShader(RenderType _rt)
{
    glm::vec3 c(0.6f,0.6f,0.6f);
    unsigned int numBones = m_rig.m_boneNameIdMapping.size();
    for(unsigned int b=0; b<numBones && b<100; b++)
    {
        glUniform3fv(m_colourAttrLoc[_rt] + b, 1, &c[0] );
    }

}

void Model::UploadBonesToShader(RenderType _rt)
{
    for(unsigned int b=0; b<m_rig.m_boneTransforms.size() && b<100; b++)
    {
        glUniformMatrix4fv(m_boneUniformLoc[_rt] + b, 1, false, &m_rig.m_boneTransforms[b][0][0]);
    }
}
