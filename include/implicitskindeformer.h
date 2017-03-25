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
    ImplicitSkinDeformer(const Mesh _origMesh,
                        const GLuint _meshVBO,
                        const std::vector<glm::mat4> &_transform);

    /// @brief Destructor.
    ~ImplicitSkinDeformer();

    //--------------------------------------------------------------------

    /// @brief Method to deform mesh.
    void Deform();

    /// @brief Method to perform LBW skinning
    /// @param _transform : a vector of joint transform matrices.
    void PerformLBWSkinning(const std::vector<glm::mat4> &_transform);

    /// @brief Method
    void PerformImplicitSkinning(const std::vector<glm::mat4> &_transform);

    //--------------------------------------------------------------------

    void AddComposedField(std::shared_ptr<ComposedField> _composedField);

    void AddFieldFunction(std::shared_ptr<FieldFunction> _fieldFunc);

    void AddCompositionOp(std::shared_ptr<CompositionOp> _compOp);

    //--------------------------------------------------------------------

    void GenerateGlobalFieldFunction(const std::vector<Mesh> &_meshParts,
                                     const std::vector<glm::vec3> &_boneStarts,
                                     const std::vector<glm::vec3> &_boneEnds,
                                     const int _numHrbfCentres);

    //--------------------------------------------------------------------

    void SetRigidTransforms(const std::vector<glm::mat4> &_transforms);

    //--------------------------------------------------------------------


    void SimpleEval(std::vector<float> &_output,
                    const std::vector<glm::vec3> &_samplePoints,
                    const std::vector<glm::mat4> &_transform);

    GlobalFieldFunction &GetGlocalFieldFunc();

private:

    //---------------------------------------------------------------------
    // Private Methods
    /// @brief Method to get the device side pointer to deformed mesh vertices for use in CUDA kernels
    glm::vec3 *GetMeshDeformedDevicePtr();

    /// @brief Method to release the device side pointer so OpenGL can use the VBO holding the deformed mesh vertices.
    void ReleaseMeshDeformedDevicePtr();


    //---------------------------------------------------------------------
    // Private Attributes

    /// @brief
    std::vector<std::thread> m_threads;

    /// @brief
    cudaGraphicsResource *m_meshVBO_CUDA;

    /// @brief
    glm::vec3 *d_meshDeformedPtr;

    /// @brief
    glm::vec3 *d_meshOrigPtr;

    /// @brief
    glm::mat4 *d_transformPtr;

    /// @brief
    unsigned int *d_boneIdPtr;

    /// @brief
    float *d_weightPtr;

    /// @brief
    bool m_meshDeformedMapped;

    /// @brief
    int m_numVerts;

    GlobalFieldFunction m_globalFieldFunction;

    bool m_init;

};

#endif // IMPLICITSKINDEFORMER_H
