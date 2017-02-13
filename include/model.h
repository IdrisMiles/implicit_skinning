#ifndef MODEL_H
#define MODEL_H

#include <GL/glew.h>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>

// GLM
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "include/rig.h"
#include "include/mesh.h"

#include "include/hrbf/hrbf_core.h"
#include "include/hrbf/hrbf_phi_funcs.h"
#include "fieldfunction.h"

#include "include/machingcube/MachingCube.h"


enum RenderType { SKINNED = 0, RIG, ISO_SURFACE, NUMRENDERTYPES };
typedef HRBF_fit<float, 3, Rbf_pow3<float> > HRBF;

class Model
{
public:


    Model();
    ~Model();
    void Load(const std::string &_mesh);
    void DrawMesh();
    void DrawRig();
    void Animate(const float _animationTime);

    void ToggleWireframe();

    void CreateShaders();
    void DeleteShaders();

    void CreateVAOs();
    void DeleteVAOs();
    void UpdateVAOs();


    void SetLightPos(const glm::vec3 &_lightPos);
    void SetModelMatrix(const glm::mat4 &_modelMat);
    void SetNormalMatrix(const glm::mat3 &_normMat);
    void SetViewMatrix(const glm::mat4 &_viewMat);
    void SetProjectionMatrix(const glm::mat4 &_projMat);

    void UploadBoneColoursToShader(RenderType _rt);
    void UploadBonesToShader(RenderType _rt);


    //-------------------------------------------------------------------
    void UpdateImplicitSurface(int xRes = 32,
                               int yRes = 32,
                               int zRes = 32,
                               float dim = 800.0f,
                               float xScale = 800.0f,
                               float yScale = 800.0f,
                               float zScale = 800.0f);


    void GenerateMeshParts();
    void GenerateFieldFunctions();
    float EvaluateDistanceFunctions(const glm::vec3 &_x);
    float RemapDistanceToFieldFunction(const float _d, const float _r);
    float EvaluateFieldFunctions(const glm::vec3 &_x);
    float CompositionOperator(const float f1, const float f2, const float d);
    float ThetaFunction(const float alpha);
    float EvaluateGlobalFieldFunction(const glm::vec3 &_x);


    //-------------------------------------------------------------------
    // Attributes
    Rig m_rig;
    Mesh m_mesh;
    Mesh m_rigMesh;

    std::vector<Mesh> m_meshParts;
    std::vector<Mesh> m_meshPartsHRBFCentres;
    std::vector<Mesh> m_meshPartsIsoSurface;
    std::vector<FieldFunction> m_fieldFunctions;

    Mesh m_meshIsoSurface;

    MachingCube m_polygonizer;

    bool m_wireframe;
    bool m_initGL;

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

    // VAO and BO's for iso-surface
    std::vector<std::shared_ptr<QOpenGLVertexArrayObject>> m_meshPartIsoVAO;
    std::vector<std::shared_ptr<QOpenGLBuffer>> m_meshPartIsoVBO;
    std::vector<std::shared_ptr<QOpenGLBuffer>> m_meshPartIsoNBO;

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

    QOpenGLShaderProgram* m_shaderProg[NUMRENDERTYPES];
};

#endif // MODEL_H
