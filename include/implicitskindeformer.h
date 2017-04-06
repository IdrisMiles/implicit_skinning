#ifndef IMPLICITSKINDEFORMER_H
#define IMPLICITSKINDEFORMER_H

#include <thread>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "mesh.h"
#include "ScalarField/globalfieldfunction.h"


/// @author Idris Miles
/// @version 1.0


/// @class ImplicitSkinDeformer
/// @brief This class deforms a mesh using the implicit skinning technique.
class ImplicitSkinDeformer
{
public:
    /// @brief Constructor.
    ImplicitSkinDeformer();

    /// @brief Destructor.
    ~ImplicitSkinDeformer();

    //--------------------------------------------------------------------

    void AttachMesh(const Mesh _origMesh,
                    const GLuint _meshVBO,
                    const std::vector<glm::mat4> &_transform);

    void GenerateGlobalFieldFunction(const std::vector<Mesh> &_meshParts,
                                     const std::vector<glm::vec3> &_boneStarts,
                                     const std::vector<glm::vec3> &_boneEnds,
                                     const int _numHrbfCentres);

    //--------------------------------------------------------------------

    /// @brief Method to deform mesh.
    void Deform();

    /// @brief Method to perform LBW skinning
    /// @param _transform : a vector of joint transform matrices.
    void PerformLBWSkinning();

    /// @brief Method
    void PerformImplicitSkinning();

    //--------------------------------------------------------------------

    void AddComposedField(std::shared_ptr<ComposedField> _composedField);

    void AddFieldFunction(std::shared_ptr<FieldFunction> _fieldFunc);

    void AddCompositionOp(std::shared_ptr<CompositionOp> _compOp);

    //--------------------------------------------------------------------

    void SetRigidTransforms(const std::vector<glm::mat4> &_transforms);

    //--------------------------------------------------------------------


    void EvalGlobalField(std::vector<float> &_output, const std::vector<glm::vec3> &_samplePoints);

    void EvalGlobalFieldInCube(std::vector<float> &_output, const int dim, const float scale);

    GlobalFieldFunction &GetGlocalFieldFunc();

private:

    //---------------------------------------------------------------------
    // Private Methods
    /// @brief
    void EvalGlobalFieldCPU(std::vector<float> &_output, const std::vector<glm::vec3> &_samplePoints);

    /// @brief
    void EvalGlobalFieldGPU(std::vector<float> &_output, const std::vector<glm::vec3> &_samplePoints);


    /// @brief
    void EvalFieldInCubeCPU(std::vector<float> &_output, const int dim, const float scale);

    /// @brief
    void EvalFieldInCubeGPU(std::vector<float> &_output, const int dim, const float scale);


    //--------------------------------------------------------------------

    void InitMeshCudaMem(const Mesh _origMesh, const GLuint _meshVBO, const std::vector<glm::mat4> &_transform);

    void InitFieldCudaMem();

    //--------------------------------------------------------------------

    /// @brief Method to get the device side pointer to deformed mesh vertices for use in CUDA kernels
    glm::vec3 *GetMeshDeformedDevicePtr();

    /// @brief Method to release the device side pointer so OpenGL can use the VBO holding the deformed mesh vertices.
    void ReleaseMeshDeformedDevicePtr();



    //---------------------------------------------------------------------
    //---------------------------------------------------------------------
    // Private Attributes

    //---------------------------------------------------------------------
    // CPU data
    /// @brief
    cudaGraphicsResource *m_meshVBO_CUDA;

    /// @brief
    std::vector<std::thread> m_threads;

    /// @brief
    bool m_meshDeformedMapped;

    /// @brief
    int m_numVerts;
    uint m_numFields;
    uint m_numCompOps;
    uint m_numCompFields;

    /// @brief
    GlobalFieldFunction m_globalFieldFunction;

    /// @brief
    bool m_initMeshCudaMem;

    /// @brief
    bool m_initFieldCudaMem;

    /// @brief
    bool m_initGobalFieldFunc;


    //---------------------------------------------------------------------
    // GPU data
    /// @brief
    glm::vec3 *d_meshDeformedPtr;

    /// @brief
    glm::vec3 *d_meshOrigPtr;

    /// @brief
    int *d_oneRingIdPtr;

    /// @brief
    int *d_numNeighsPerVertPtr;

    /// @brief
    int *d_oneRingScatterAddrPtr;

    float *d_centroidWeights;

    /// @brief
    glm::vec3 *d_oneRingVertPtr;

    /// @brief
    float *d_origVertIsoPtr;

    /// @brief
    float *d_newVertIsoPtr;

    /// @brief
    glm::mat4 *d_transformPtr;
    int m_numTransforms;

    /// @brief
    glm::mat4 *d_textureSpacePtr;

    /// @brief
    cudaTextureObject_t *d_fieldsPtr;

    /// @brief
    cudaTextureObject_t *d_gradPtr;

    /// @brief
    cudaTextureObject_t *d_compOpPtr;

    /// @brief
    ComposedFieldCuda *d_compFieldPtr;

    /// @brief
    unsigned int *d_boneIdPtr;

    /// @brief
    float *d_weightPtr;


};

#endif // IMPLICITSKINDEFORMER_H
