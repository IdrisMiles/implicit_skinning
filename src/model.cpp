#include "include/model.h"
#include "include/modelloader.h"

#include <iostream>

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
    }
}

void Model::Load(const std::string &_file)
{
    ModelLoader::LoadModel(this, _file);

    CreateShaders();
    CreateVAOs();
    UpdateVAOs();

}

void Model::DrawMesh()
{
    if(!m_initGL)
    {
        CreateVAOs();
        UpdateVAOs();
    }
    else
    {
        m_shaderProg[SKINNED]->bind();
        glUniformMatrix4fv(m_projMatrixLoc[SKINNED], 1, false, &m_projMat[0][0]);
        glUniformMatrix4fv(m_mvMatrixLoc[SKINNED], 1, false, &(m_modelMat*m_viewMat)[0][0]);
        glm::mat3 normalMatrix =  glm::inverse(glm::mat3(m_modelMat));
        glUniformMatrix3fv(m_normalMatrixLoc[SKINNED], 1, true, &normalMatrix[0][0]);
        glUniform3fv(m_colourLoc[SKINNED], 1, &m_mesh.m_colour[0]);

        m_meshVAO[SKINNED].bind();
        glPolygonMode(GL_FRONT_AND_BACK, m_wireframe?GL_LINE:GL_FILL);
        glDrawElements(GL_TRIANGLES, 3*m_mesh.m_meshTris.size(), GL_UNSIGNED_INT, &m_mesh.m_meshTris[0]);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        m_meshVAO[SKINNED].release();

        m_shaderProg[SKINNED]->release();
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
        glDrawArrays(GL_LINES, 0, m_mesh.m_rigVerts.size());
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
    if(m_shaderProg[SKINNED]->link())
    {

    }
    else
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
    if(m_shaderProg[RIG]->link())
    {

    }
    else
    {
        std::cout<<m_shaderProg[SKINNED]->log().toStdString()<<"\n";
    }

    m_shaderProg[RIG]->bind();
    m_projMatrixLoc[RIG] = m_shaderProg[RIG]->uniformLocation("projMatrix");
    m_mvMatrixLoc[RIG] = m_shaderProg[RIG]->uniformLocation("mvMatrix");
    m_normalMatrixLoc[RIG] = m_shaderProg[RIG]->uniformLocation("normalMatrix");
    m_shaderProg[RIG]->release();


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


        std::cout<<"SKINNED | vert Attr loc:\t"<<m_vertAttrLoc[SKINNED]<<"\n";
        std::cout<<"SKINNED | norm Attr loc:\t"<<m_normAttrLoc[SKINNED]<<"\n";
        std::cout<<"SKINNED | boneID Attr loc:\t"<<m_boneIDAttrLoc[SKINNED]<<"\n";
        std::cout<<"SKINNED | boneWei Attr loc:\t"<<m_boneWeightAttrLoc[SKINNED]<<"\n";
        std::cout<<"SKINNED | Bones Unif loc:\t"<<m_boneUniformLoc[SKINNED]<<"\n";
        std::cout<<"SKINNED | Bones Colour Unif loc:\t"<<m_colourAttrLoc[SKINNED]<<"\n";


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
    m_shaderProg[SKINNED]->bind();

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



    //--------------------------------------------------------------------------------------
    // Rigged mesh
    m_shaderProg[RIG]->bind();

    m_meshVAO[RIG].bind();

    // Setup our vertex buffer object.
    m_meshVBO[RIG].bind();
    m_meshVBO[RIG].allocate(&m_mesh.m_rigVerts[0], m_mesh.m_rigVerts.size() * sizeof(glm::vec3));
    glEnableVertexAttribArray(m_vertAttrLoc[RIG]);
    glVertexAttribPointer(m_vertAttrLoc[RIG], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
    m_meshVBO[RIG].release();

    // Set up our Rig joint colour buffer object
    m_meshCBO[RIG].bind();
    m_meshCBO[RIG].allocate(&m_mesh.m_rigJointColours[0], m_mesh.m_rigJointColours.size() * sizeof(glm::vec3));
    glEnableVertexAttribArray(m_colourAttrLoc[RIG]);
    glVertexAttribPointer(m_colourAttrLoc[RIG], 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), 0);
    m_meshCBO[RIG].release();

    // Set up vertex bone weighting buffer object
    m_meshBWBO[RIG].bind();
    m_meshBWBO[RIG].allocate(&m_mesh.m_rigBoneWeights[0], m_mesh.m_rigBoneWeights.size() * sizeof(VertexBoneData));
    glEnableVertexAttribArray(m_boneIDAttrLoc[RIG]);
    glVertexAttribIPointer(m_boneIDAttrLoc[RIG], MaxNumBlendWeightsPerVertex, GL_UNSIGNED_INT, sizeof(VertexBoneData), (const GLvoid*)0);
    glEnableVertexAttribArray(m_boneWeightAttrLoc[RIG]);
    glVertexAttribPointer(m_boneWeightAttrLoc[RIG], MaxNumBlendWeightsPerVertex, GL_FLOAT, GL_FALSE, sizeof(VertexBoneData), (const GLvoid*) (MaxNumBlendWeightsPerVertex*sizeof(unsigned int)));
    m_meshBWBO[RIG].release();

    m_meshVAO[RIG].release();

    m_shaderProg[RIG]->release();
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
        glUniformMatrix4fv(m_boneUniformLoc[_rt] + b, 1, true, &m_rig.m_boneTransforms[b][0][0]);
    }
}
