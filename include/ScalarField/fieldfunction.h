#ifndef FIELDFUNCTION_H
#define FIELDFUNCTION_H

#include <glm/glm.hpp>
#include <utility>

#include "Hrbf/hrbf_core.h"
#include "Hrbf/hrbf_phi_funcs.h"

#include "ScalarField/field1d.h"

#include <cuda.h>
#include "cudatexture.h"




typedef HRBF_fit<float, 3, Rbf_pow3<float> > DistanceField;

class FieldFunction
{
public:
    FieldFunction();
    ~FieldFunction();

    void Fit(const std::vector<glm::vec3>& points,
             const std::vector<glm::vec3>& normals,
             const float _r = 1.0f);

    /// @brief Method to precompute field values and store them in m_field attribute.
    void PrecomputeField(const unsigned int _dim = 32, const float _scale = 8.0f);

    /// @brief Method to set the support radius in order to remap the distance field
    /// to a compact field function with range [0:1]
    void SetSupportRadius(const float _r);

    void SetTransform(glm::mat4 _transform);

    glm::mat4 GetTransform() const;

    glm::mat4 GetTextureSpaceTransform() const;

    float Eval(const glm::vec3& _x);

    float EvalDist(const glm::vec3& x);

    glm::vec3 Grad(const glm::vec3& x);

    cudaTextureObject_t &GetFieldFunc3DTexture();
    cudaTextureObject_t &GetFieldGrad3DTexture();



private:

    /// @brief Method to remap distance field values to a compactly supported field function [0:1]
    /// @param float _distValue value we need to remap to be between [0:1]
    /// @return float field value between [0:1]
    float Remap(float _df);

    glm::vec3 TransformSpace(glm::vec3 _x);

    bool Equiv(glm::vec3 _a, glm::vec3 _b);

    bool m_fit;

    bool m_precomputedGPU;

    bool m_precomputedCPU;

    /// @brief Attribute used for remapping distance field to compactly supported field function.
    float m_supportRad;

    /// @brief
    glm::mat4 m_transform;

    /// @brief A matrix to transform from world space to texture space.
    /// Used for texture lookup.
    glm::mat4 m_textureSpaceTransform;

    /// @brief an HRBF distance field generator
    DistanceField m_distanceField;

    /// @brief A field object to store precomputed field value,
    /// improves performance to interpolate values than compute them.
    Field3D<float> m_field;
    Field3D<glm::vec3> m_grad;

    Cuda3DTexture<float> d_field;
    Cuda3DTexture<float4> d_grad;


};

#endif // FIELDFUNCTION_H
