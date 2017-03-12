#ifndef IMPLICITSKINKERNELS_H
#define IMPLICITSKINKERNELS_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

//#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
//#endif
#include <glm/glm.hpp>
#include <glm/gtx/vector_angle.hpp>

#include <mesh.h>


class ImplicitSkinKernels
{
public:
    ImplicitSkinKernels(const Mesh _origMesh,
                        const GLuint _meshVBO,
                        const std::vector<glm::mat4> &_transform);

    ~ImplicitSkinKernels();

    void Deform();

    void PerformLBWSkinning(const std::vector<glm::mat4> &_transform);
    void PerformVertexProjection();
    void PerformTangentialRelaxation();
    void PerformLaplacianSmoothing();

private:

    //---------------------------------------------------------------------
    // Private Methods
    glm::vec3 *GetMeshDeformedPtr();

    void ReleaseMeshDeformedPtr();


    //---------------------------------------------------------------------
    // Private Attributes
    cudaGraphicsResource *m_meshVBO_CUDA;

    glm::vec3 *d_meshDeformedPtr;
    glm::vec3 *d_meshOrigPtr;
    glm::mat4 *d_transformPtr;
    unsigned int *d_boneIdPtr;
    float *d_weightPtr;

    bool m_meshDeformedMapped;
    int m_numVerts;

};

#endif // IMPLICITSKINKERNELS_H
