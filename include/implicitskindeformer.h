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
/// @date 18/04/2017


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

    /// @brief Method to attach a mesh to the deformer, and upload data to the GPU
    /// @param _origMesh : Mesh holding vertices, normals, bone weight and ids
    /// @param _meshVBO : The meshes vertex buffer object, need this to map resources so we can directly deform mesh vertices in CUDA
    /// @param _meshNBO : The meshes normal buffer object, need this to map resources so we can directly deform mesh normals in CUDA
    /// @param _transform : The rest bone transforms.
    void AttachMesh(const Mesh _origMesh,
                    const GLuint _meshVBO,
                    const GLuint _meshNBO,
                    const std::vector<glm::mat4> &_transform);

    /// @brief Method to generate the global function from the various mesh parts
    /// @param _meshParts : A vector of meshes containing individual mesh parts, for example: left lower leg, left upper leg etc..
    /// @param _boneStarts : A vector of the positions in 3D space of the start and end of the bones corresponding to the mesh parts.
    /// @param _numHrbfCentres : The numbere of Hermite Radial Basis Function (HRBF) centres used to generate the field function.
    void GenerateGlobalFieldFunction(const std::vector<Mesh> &_meshParts,
                                     const std::vector<std::pair<glm::vec3, glm::vec3>> &_boneEnds,
                                     const int _numHrbfCentres = 50);

    //--------------------------------------------------------------------

    /// @brief Method to deform mesh.
    void Deform();

    /// @brief Method to perform LBW skinning
    /// @param _transform : a vector of joint transform matrices.
    void PerformLBWSkinning();

    /// @brief Method to perform Implicit skinning
    void PerformImplicitSkinning();

    //--------------------------------------------------------------------

    /// @brief Method to update bone transforms.
    /// @brief This is seperated from the deform method as animation and visualisation maybe on different timers.
    /// @param _transforms : Updated bone transforms
    void SetRigidTransforms(const std::vector<glm::mat4> &_transforms);

    //--------------------------------------------------------------------

    /// @brief Method to evaluate the global field function at a set of given sasmple points.
    /// @param _output : The output values of the evaulated field at the sampel points.
    /// @param _samplePoints : A vector of positions in 3D space to sample the global field function.
    void EvalGlobalField(std::vector<float> &_output, const std::vector<glm::vec3> &_samplePoints);

    /// @brief Method to evaluate the global field function at regular uniform intervals within a cube
    /// @param _output : The output values of the evaulated field at the sampel points.
    /// @param res : The resolution of the cube volume to sample, number of sample points = res*res*res
    /// @param dim : The dimension of the cube volume to sample.
    void EvalGlobalFieldInCube(std::vector<float> &_output, const int res, const float dim);


private:

    //--------------------------------------------------------------------
    // Method for building up the global field

    /// @brief Method to add a composed field to the global field
    /// @param _composedField : A pointer to the composed field
    void AddComposedField(std::shared_ptr<ComposedField> _composedField);

    /// @brief Method to add a field function to the global field
    /// @param _fieldFuc : A pointer to a field function
    void AddFieldFunction(std::shared_ptr<FieldFunction> _fieldFunc);

    /// @brief Method to add a composition operator to the global field
    /// @param _compOp : A pointer to the composition operator
    void AddCompositionOp(std::shared_ptr<CompositionOp> _compOp);


    //--------------------------------------------------------------------
    // methods for evaluating the global field

    /// @brief Private method to evaulate the global field on the CPU
    /// @param _output : The output values of the evaulated field at the sampel points.
    /// @param _samplePoints : A vector of positions in 3D space to sample the global field function.
    void EvalGlobalFieldCPU(std::vector<float> &_output, const std::vector<glm::vec3> &_samplePoints);

    /// @brief Private method to evaluate the global field on the GPU
    /// @param _output : The output values of the evaulated field at the sampel points.
    /// @param _samplePoints : A vector of positions in 3D space to sample the global field function.
    void EvalGlobalFieldGPU(std::vector<float> &_output, const std::vector<glm::vec3> &_samplePoints);


    /// @brief Private method to evaluate the global field at regular uniform intervals within a cube on the CPU
    /// @param _output : The output values of the evaulated field at the sampel points.
    /// @param res : The resolution of the cube volume to sample, number of sample points = res*res*res
    /// @param dim : The dimension of the cube volume to sample.
    void EvalFieldInCubeCPU(std::vector<float> &_output, const int res, const float dim);

    /// @brief Private method to evaluate the global field at regular uniform intervals within a cube on the GPU
    /// @param _output : The output values of the evaulated field at the sampel points.
    /// @param res : The resolution of the cube volume to sample, number of sample points = res*res*res
    /// @param dim : The dimension of the cube volume to sample.
    void EvalFieldInCubeGPU(std::vector<float> &_output, const int res, const float dim);


    //--------------------------------------------------------------------
    // Methods for initlaising the deformer

    /// @brief Method to initialise the Iso values of each mesh vertex in the global field
    void InitialiseIsoValues();

    /// @brief Method to initialise CUDA memory that will hold the mesh to be deformed
    /// @param _origMesh : The mesh we which to upload to the GPU
    /// @param _memshVBO : The meshes vertex buffer object which we need to map resources inorder to drectly deform the meshes vertices on the GPU
    /// @param _memshNBO : The meshes norrmal buffer object which we need to map resources inorder to drectly deform the meshes normals on the GPU
    /// param _transforms : The default bone transforms.
    void InitMeshCudaMem(const Mesh _origMesh, const GLuint _meshVBO, const GLuint _meshNBO, const std::vector<glm::mat4> &_transform);

    /// @brief Method to initialise CUDA memory that will hold the global field
    void InitFieldCudaMem();

    /// @brief Method to clean up CUDA memory allocated for storing the mesh
    void DestroyMeshCudaMem();

    /// @brief Method to clean up CUDA memory allocated for storing the global field
    void DestroyFieldCudaMem();


    //--------------------------------------------------------------------
    // Methods for accessing GPU resources within the deformer

    /// @brief Method to get the device side pointer to deformed mesh vertices for use in CUDA kernels.
    /// @brief We must map resources before directly accessing it.
    glm::vec3 *GetDeformedMeshVertsDevicePtr();

    /// @brief Method to release the device side pointer so OpenGL can use the VBO holding the deformed mesh vertices.
    /// @brief We must unmap resources once we are finished using it.
    void ReleaseDeformedMeshVertsDevicePtr();

    /// @brief Method to get the device side pointer to deformed mesh normals for use in CUDA kernels
    /// @brief We must map resources before directly accessing it.
    glm::vec3 *GetDeformedMeshNormsDevicePtr();

    /// @brief Method to release the device side pointer so OpenGL can use the NBO holding the deformed mesh normals.
    /// @brief We must unmap resources once we are finished using it.
    void ReleaseDeformedMeshNormsDevicePtr();



    //---------------------------------------------------------------------
    // CPU Attributes

    /// @brief The CUDA graphics resource to map the mesh vertex buffer object to a pointer that can be used within a CUDA kernel
    cudaGraphicsResource *m_meshVBO_CUDA;

    /// @brief The CUDA graphics resource to map the mesh normal buffer object to a pointer that can be used within a CUDA kernel
    cudaGraphicsResource *m_meshNBO_CUDA;

    /// @brief A sort of thread pool to start up some threads quickly for some data processing
    std::vector<std::thread> m_threads;

    /// @brief A boolean to check whether the m_meshVBO_CUDA object has been mapped.
    bool m_deformedMeshVertsMapped;

    /// @brief A boolean to check whether the m_meshNBO_CUDA object has been mapped.
    bool m_deformedMeshNormsMapped;

    /// @brief The number of vertices in the mesh that has been attached for deformation.
    int m_numVerts;

    /// @brief The number of primitive fields that our global field is composed of.
    uint m_numFields;
    uint m_numCompOps;
    uint m_numCompFields;
    int m_numTransforms;
    glm::vec3 m_minBBox;
    glm::vec3 m_maxBBox;

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
    glm::vec3 *d_deformedMeshVertsPtr;

    /// @brief
    glm::vec3 *d_origMeshVertsPtr;

    /// @brief
    glm::vec3 *d_deformedMeshNormsPtr;

    /// @brief
    glm::vec3 *d_origMeshNormsPtr;

    /// @brief
    int *d_oneRingIdPtr;

    /// @brief
    int *d_numNeighsPerVertPtr;

    /// @brief
    int *d_oneRingScatterAddrPtr;

    float *d_centroidWeightsPtr;

    /// @brief
    glm::vec3 *d_oneRingVertPtr;

    /// @brief
    float *d_origVertIsoPtr;

    /// @brief
    float *d_newVertIsoPtr;

    /// @brief
    glm::vec3 *d_vertIsoGradPtr;

    /// @brief
    glm::mat4 *d_transformPtr;

    /// @brief
    glm::mat4 *d_textureSpacePtr;

    /// @brief
    cudaTextureObject_t *d_fieldsPtr;

    /// @brief
    cudaTextureObject_t *d_compOpPtr;

    /// @brief
    cudaTextureObject_t *d_thetaPtr;

    /// @brief
    ComposedFieldCuda *d_compFieldPtr;

    /// @brief
    unsigned int *d_boneIdPtr;

    /// @brief
    float *d_weightPtr;


};

#endif // IMPLICITSKINDEFORMER_H
