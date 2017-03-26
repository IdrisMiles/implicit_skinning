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


Model::Model()
{
    m_wireframe = false;
    m_initGL = false;
}

Model::~Model()
{

    if(m_initGL)
    {
        DeleteVAOs();
        DeleteImplicitSkinner();
    }
}

void Model::Load(const std::string &_file)
{
    ModelLoader::LoadModel(this, _file);


    //--------------------------------------------------

    CreateShaders();
    CreateVAOs();
    UpdateVAOs();


    //--------------------------------------------------

    InitImplicitSkinner();
    GenerateMeshParts();
    GenerateFieldFunctions();
}


void Model::GenerateMeshParts()
{
    unsigned int numParts = m_rig.m_boneNameIdMapping.size();
    m_meshParts.resize(numParts);


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
    std::vector<glm::vec3> boneStarts;
    std::vector<glm::vec3> boneEnds;
    for(int i=0; i<m_meshParts.size(); i++)
    {
        boneStarts.push_back(m_rigMesh.m_meshVerts[i*2]);
        boneEnds.push_back(m_rigMesh.m_meshVerts[(i*2) + 1]);
    }
    m_implicitSkinner->GenerateGlobalFieldFunction(m_meshParts, boneStarts, boneEnds, 50);
}


//---------------------------------------------------------------------------------

void Model::DeformSkin()
{
    m_implicitSkinner->PerformLBWSkinning(m_rig.m_boneTransforms);
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
    // Generate sample points
    std::vector<glm::vec3> samplePoints(xRes*yRes*zRes);
    for(int z=0;z<zRes;z++)
    {
        for(int y=0;y<yRes;y++)
        {
            for(int x=0;x<xRes;x++)
            {
                samplePoints[(z*yRes*xRes) + (y*xRes) + x] = glm::vec3(dim*((((float)x/zRes)*2.0f)-1.0f),
                                                                       dim*((((float)y/yRes)*2.0f)-1.0f),
                                                                       dim*((((float)z/xRes)*2.0f)-1.0f));
            }
        }
    }


    // Evaluate field
    std::vector<float> f;
    m_implicitSkinner->EvalField(f, samplePoints);


    // Polygonize scalar field using maching cube
    m_polygonizer.Polygonize(m_meshIsoSurface.m_meshVerts, m_meshIsoSurface.m_meshNorms, &f[0], 0.5f, xRes, yRes, zRes, xScale, yScale, zScale);
}


void Model::DrawMesh()
{

    static float angle = 0.0f;
    angle+=0.1f;
    if(!m_initGL)
    {
        CreateVAOs();
        UpdateVAOs();
        InitImplicitSkinner();
    }
    else
    {
        //-------------------------------------------------------------------------------------
        // Skin our mesh
        DeformSkin();


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


        // upload new verts
        glUniform3fv(m_colourLoc[ISO_SURFACE], 1, &m_meshIsoSurface.m_colour[0]);// Setup our vertex buffer object.
        m_meshVBO[ISO_SURFACE].bind();
        m_meshVBO[ISO_SURFACE].allocate(&m_meshIsoSurface.m_meshVerts[0], m_meshIsoSurface.m_meshVerts.size() * sizeof(glm::vec3));
        glEnableVertexAttribArray(m_vertAttrLoc[ISO_SURFACE]);
        glVertexAttribPointer(m_vertAttrLoc[ISO_SURFACE], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshVBO[ISO_SURFACE].release();


        // upload new normals
        m_meshNBO[ISO_SURFACE].bind();
        m_meshNBO[ISO_SURFACE].allocate(&m_meshIsoSurface.m_meshNorms[0], m_meshIsoSurface.m_meshNorms.size() * sizeof(glm::vec3));
        glEnableVertexAttribArray(m_normAttrLoc[ISO_SURFACE]);
        glVertexAttribPointer(m_normAttrLoc[ISO_SURFACE], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
        m_meshNBO[ISO_SURFACE].release();


        // Draw marching cube of isosurface
        m_meshVAO[ISO_SURFACE].bind();
        glPolygonMode(GL_FRONT_AND_BACK, m_wireframe?GL_FILL:GL_FILL);
        glDrawArrays(GL_TRIANGLES, 0, m_meshIsoSurface.m_meshVerts.size());
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        m_meshVAO[ISO_SURFACE].release();

        m_shaderProg[ISO_SURFACE]->release();
    }

}

void Model::DrawRig()
{
    if(!m_initGL)
    {
        CreateVAOs();
        UpdateVAOs();
        InitImplicitSkinner();
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

    m_implicitSkinner->SetRigidTransforms(m_rig.m_boneTransforms);
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
}


void Model::InitImplicitSkinner()
{
    m_implicitSkinner = new ImplicitSkinDeformer(m_mesh, m_meshVBO[SKINNED].bufferId(), m_rig.m_boneTransforms);
}

void Model::DeleteImplicitSkinner()
{
    if(m_implicitSkinner != nullptr)
    {
        delete m_implicitSkinner;
        m_implicitSkinner = nullptr;
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
