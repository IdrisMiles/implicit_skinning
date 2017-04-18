#ifndef MODEL_H
#define MODEL_H

#include <thread>

#include <GL/glew.h>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>

// GLM
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "include/rig.h"
#include "include/mesh.h"

#include "ScalarField/Hrbf/hrbf_core.h"
#include "ScalarField/Hrbf/hrbf_phi_funcs.h"
#include "ScalarField/fieldfunction.h"
#include "ScalarField/globalfieldfunction.h"

#include "Machingcube/MachingCube.h"

#include "implicitskindeformer.h"


/// @author Idris Miles
/// @version 1.0
/// @data 18/04/2017


enum RenderType { SKINNED = 0, RIG, ISO_SURFACE, NUMRENDERTYPES };
typedef HRBF_fit<float, 3, Rbf_pow3<float> > HRBF;


/// @class Model
/// @brief A model, this holds the mesh, animated rig, and skin deformer, it also handles the rendering of itself.
class Model
{

public:

    /// @brief constructor
    Model();

    /// @brief destructor
    ~Model();

    /// @brief Method to laod a model from file
    /// @param _mesh : The file name of the model we wish to load
    void Load(const std::string &_mesh);

    /// @brief Method to draw the skinned animated model
    void DrawMesh();

    /// @brief Method to draw the animated rig
    void DrawRig();

    /// @brief Method to animate the model
    /// @param _animationTime : the time the animation should be a
    void Animate(const float _animationTime);

    /// @brief Method to toggle rendering the skinned mesh in wireframe or filled
    void ToggleWireframe();

    /// @brief Method to toggle rendering of the skinned mesh
    void ToggleSkinnedSurface();

    /// @brief Method to toggle performing implicit or linear blend weight skinning
    void ToggleSkinnedImplicitSurface();

    /// @brief Method to toggle rendering the iso surface of the global field from the implicit deformer
    void ToggleIsoSurface();

    /// @brief Method to set light position
    void SetLightPos(const glm::vec3 &_lightPos);

    /// @brief Method to set the model matrix
    void SetModelMatrix(const glm::mat4 &_modelMat);

    /// @brief Method to set the normal matrix
    void SetNormalMatrix(const glm::mat3 &_normMat);

    /// @brief Method to set the view matrix
    void SetViewMatrix(const glm::mat4 &_viewMat);

    /// @brief Method to set the projection matrix
    void SetProjectionMatrix(const glm::mat4 &_projMat);

    /// @brief Method to get the Rig
    Rig &GetRig();

    /// @brief Method to get the model mesh
    Mesh &GetMesh();

    /// @brief Method to get the Rig Mesh
    Mesh &GetRigMesh();


private:

    /// @brief Method to create shaders
    void CreateShaders();

    /// @brief Method to delete shaders
    void DeleteShaders();

    /// @brief Method to create all necessary Vertex array objects and buffer objects
    void CreateVAOs();

    /// @brief Method to delete VAO's and BO's
    void DeleteVAOs();

    /// @brief Method to update contents of the VAO and BO's
    void UpdateVAOs();


    /// @brief Method to upload bone colours to shader
    void UploadBoneColoursToShader(RenderType _rt);

    /// @brief Method to upload bone transforms to shader
    void UploadBonesToShader(RenderType _rt);

    /// @brief Method to update the iso surface for rendering
    void UpdateIsoSurface(int xRes = 32,
                          int yRes = 32,
                          int zRes = 32,
                          float dim = 800.0f,
                          float xScale = 800.0f,
                          float yScale = 800.0f,
                          float zScale = 800.0f);

    /// @brief Method to generate mesh parts used for the implicit skinner
    void GenerateMeshParts(std::vector<Mesh> &_meshParts);

    /// @brief Method to initialise the implicit skinner
    void InitImplicitSkinner();

    /// @brief Method to delete and clean up after the implicit skinner
    void DeleteImplicitSkinner();

    /// @brief Method to deform the skin using the implicit skin deformer
    void DeformSkin();


    //-------------------------------------------------------------------
    // Attributes

    /// @brief Implicit skinner used to deform the mesh of our model
    ImplicitSkinDeformer *m_implicitSkinner;

    /// @brief a mini thread pool used to speed up data processing
    std::vector<std::thread> m_threads;

    /// @brief This models rig
    Rig m_rig;

    /// @brief This models mesh
    Mesh m_mesh;

    /// @brief This models rigs mesh
    Mesh m_rigMesh;

    /// @brief The mesh for the Iso surface produced by the implicit deformer
    Mesh m_meshIsoSurface;

    ///
    MachingCube m_polygonizer;

    std::vector<float> m_meshVertIsoValues;
    std::vector<std::vector<unsigned int>> m_meshVertOneRingNeighbour;
    std::vector<std::vector<float>> m_meshVertCentroidWeights;


    bool m_wireframe;
    bool m_drawSkin;
    bool m_drawImplicitSkin;
    bool m_drawIsoSurface;
    bool m_initGL;

    glm::vec3 m_lightPos;
    glm::mat4 m_projMat;
    glm::mat4 m_viewMat;
    glm::mat4 m_modelMat;
    glm::mat3 m_normMat;

    // OpenGL VAO and BO's LBW skinned mesh
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

    QOpenGLShaderProgram* m_shaderProg[NUMRENDERTYPES];
};

#endif // MODEL_H
