#ifndef IMPLICITSKINKERNELS_H
#define IMPLICITSKINKERNELS_H

#include <cuda_runtime.h>
#include <cuda.h>

//#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
//#endif
#include <glm/glm.hpp>
#include <glm/gtx/vector_angle.hpp>



class ImplicitSkinKernels
{
public:
    ImplicitSkinKernels();

    void Deform();

    static void PerformLBWSkinning(glm::vec3 *_deformedMeshVerts,
                                   glm::vec3 *_origMeshVerts,
                                   glm::mat4 *_transform,
                                   uint *_boneId,
                                   float *_weight,
                                   uint _numVerts,
                                   uint _numBones);
    void PerformVertexProjection();
    void PerformTangentialRelaxation();
    void PerformLaplacianSmoothing();
};

#endif // IMPLICITSKINKERNELS_H
