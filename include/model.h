#ifndef MODEL_H
#define MODEL_H

#include <GL/glew.h>

#include <QOpenGLWidget>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>

// GLM
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "include/rig.h"
#include "include/mesh.h"

class Model
{
public:

    enum RenderType { SKINNED = 0, RIG = 1, NUMRENDERTYPES };

    Model();
    void Load(const std::string &_mesh);
    void DrawMesh();
    void DrawRig();
    void Animate(const float _animationTime);


    void CreateShaders();
    void DeleteShaders();

    void CreateVAOs();
    void DeleteVAOs();
    void UpdateVAOs();


    void SetLightPos(const glm::vec3 &_lightPos);
    void SetModelMatrix(const glm::mat4 &_modelMat);
    void SetNormalMatrix(const glm::mat4 &_normMat);
    void SetViewMatrix(const glm::mat4 &_viewMat);
    void SetProjectionMatrix(const glm::mat4 &_projMat);


    void UploadBoneColoursToShader(RenderType _rt);
    void UploadBonesToShader(RenderType _rt);



    // Attributes
    Rig m_rig;
    Mesh m_mesh;

    bool m_wireframe;

    glm::vec3 m_lightPos;
    glm::mat4 m_projMat;
    glm::mat4 m_viewMat;
    glm::mat4 m_modelMat;
    glm::mat3 m_normMat;

    // OpenGL VAO and BO's
    QOpenGLVertexArrayObject m_meshVAO[NUMRENDERTYPES];
    QOpenGLBuffer m_meshVBO[NUMRENDERTYPES];
    QOpenGLBuffer m_meshNBO[NUMRENDERTYPES];
    QOpenGLBuffer m_meshIBO[NUMRENDERTYPES];
    QOpenGLBuffer m_meshBWBO[NUMRENDERTYPES];
    QOpenGLBuffer m_meshCBO[NUMRENDERTYPES];

    // Shader locations
    GLuint m_vertAttrLoc[NUMRENDERTYPES];
    GLuint m_normAttrLoc[NUMRENDERTYPES];
    GLuint m_boneIDAttrLoc[NUMRENDERTYPES];
    GLuint m_boneWeightAttrLoc[NUMRENDERTYPES];
    GLuint m_boneUniformLoc[NUMRENDERTYPES];
    GLuint m_colourLoc[NUMRENDERTYPES];
    GLuint m_colourAttrLoc[NUMRENDERTYPES];
    GLuint m_projMatrixLoc[NUMRENDERTYPES];
    GLuint m_mvMatrixLoc[NUMRENDERTYPES];
    GLuint m_normalMatrixLoc[NUMRENDERTYPES];
    GLuint m_lightPosLoc[NUMRENDERTYPES];

    QOpenGLShaderProgram m_shaderProg[NUMRENDERTYPES];
};

#endif // MODEL_H
