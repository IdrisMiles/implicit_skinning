#ifndef FIELDFUNCTION_H
#define FIELDFUNCTION_H

#include <glm/glm.hpp>

#include "Hrbf/hrbf_core.h"
#include "Hrbf/hrbf_phi_funcs.h"

#include "ScalarField/field1d.h"

typedef HRBF_fit<float, 3, Rbf_pow3<float> > DistanceField;

class FieldFunction
{
public:
    FieldFunction(glm::mat4 _transform = glm::mat4(1.0f));
    ~FieldFunction();

    void Fit(const std::vector<glm::vec3>& points,
             const std::vector<glm::vec3>& normals,
             const float _r = 1.0f);

    /// @brief Method to precompute field values and store them in m_field attribute.
    void PrecomputeField();

    void SetR(const float _r);

    void SetTransform(glm::mat4 _transform);

    float Eval(const glm::vec3& x);
    float EvalDist(const glm::vec3& x);

    glm::vec3 Grad(const glm::vec3& x);


private:

    /// @brief Method to remap distance field values to a compactly supported field function [0:1]
    /// @param float _distValue value we need to remap to be between [0:1]
    /// @return float field value between [0:1]
    float Remap(float _df);

    glm::vec3 TransformSpace(glm::vec3 _x);

    /// @brief Attribute used for remapping distance field to compactly supported field function.
    float r;

    /// @brief
    glm::mat4 m_transform;

    /// @brief an HRBF distance field generator
    DistanceField m_distanceField;

    /// @brief A field object to store precomputed field value,
    /// improves performance to interpolate values than compute them.
    Field1D m_field;

};

#endif // FIELDFUNCTION_H
