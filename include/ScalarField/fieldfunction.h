#ifndef FIELDFUNCTION_H
#define FIELDFUNCTION_H

//--------------------------------------------------------------------------------
// includes

#include <glm/glm.hpp>
#include <utility>

#include "Hrbf/hrbf_core.h"
#include "Hrbf/hrbf_phi_funcs.h"

#include <cuda.h>

#include "Texture3DCuda.h"
#include "Texture3DCpu.h"


//--------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @data 18/04/2017
//--------------------------------------------------------------------------------


/// @typedef DistanceField
typedef HRBF_fit<float, 3, Rbf_pow3<float> > DistanceField;


/// @class FieldFunction
/// @brief This class
class FieldFunction
{
public:
    /// @brief constructor
    FieldFunction();
    /// @brief destructor
    ~FieldFunction();

    /// @brief Method to fit the field function given a set of points and normals
    /// @param points : HRBF centres used to generate field function
    /// @param normals : HRBF normals used to generate field function
    /// @param _r : support radius of field function
    void Fit(const std::vector<glm::vec3>& points,
             const std::vector<glm::vec3>& normals,
             const float _r = 1.0f);

    /// @brief Method to precompute field values and store them in m_field attribute.
    void PrecomputeField(const unsigned int _res = 32, const float _dim = 8.0f);

    /// @brief Method to set the support radius in order to remap the distance field
    /// to a compact field function with range [0:1]
    void SetSupportRadius(const float _r);

    /// @brief Method to set transform for this field
    void SetTransform(glm::mat4 _transform);

    /// @brief Method to get the transform associated with this field
    glm::mat4 GetTransform() const;

    /// @brief Method to get the texture space transform associated with this field
    glm::mat4 GetTextureSpaceTransform() const;

    /// @brief Method to evaluate this field
    /// @param _x : sample point
    float Eval(const glm::vec3& _x);

    /// @brief Method to evaluate the underlying distance field function
    /// @param _x : sample point
    float EvalDist(const glm::vec3& x);

    /// @brief Method to evaluate the gradient of the field
    /// @param _x : sample point
    glm::vec3 Grad(const glm::vec3& x);

    /// @brief Method to Get the cuda texture object holding the field function
    cudaTextureObject_t &GetFieldFuncCudaTextureObject();

    /// @brief Method to Get the cuda texture object holding the gradiet of the field function
    cudaTextureObject_t &GetFieldGradCudaTextureObject();



private:

    /// @brief Method to remap distance field values to a compactly supported field function [0:1]
    /// @param float _distValue value we need to remap to be between [0:1]
    /// @return float field value between [0:1]
    float Remap(float _df);

    /// @brief Method to apply the transform associated to this field to point_x
    /// @param _x : sample point
    /// @return glm::vec3 newly transformed point
    glm::vec3 TransformSpace(glm::vec3 _x);

    /// @brief boolean too check whether field function has been fitted yet
    bool m_fit;

    /// @brief boolean to check if the GPU texture has been precomputed
    bool m_precomputedGPU;

    /// @brief boolean to check if the CPU texture has been precomputed
    bool m_precomputedCPU;

    /// @brief Attribute used for remapping distance field to compactly supported field function.
    float m_supportRad;

    /// @brief Transform associated with this field
    glm::mat4 m_transform;

    /// @brief A matrix to transform from world space to texture space.
    /// Used for texture lookup.
    glm::mat4 m_textureSpaceTransform;

    /// @brief an HRBF distance field generator
    DistanceField m_distanceField;

    /// @brief A CPU based 3D texture to store precomputed field value
    Texture3DCpu<float> m_field;

    /// @brief A CPU based 3D texture to store precomputed gradient value
    Texture3DCpu<glm::vec3> m_grad;

    /// @brief A GPU based 3D texture to store precomputed field value
    Texture3DCuda<float> d_field;

    /// @brief A GPU based 3D texture to store precomputed gradient value
    Texture3DCuda<float4> d_grad;


};

#endif // FIELDFUNCTION_H
