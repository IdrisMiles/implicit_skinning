#ifndef FIELDFUNCTION_H
#define FIELDFUNCTION_H


#include "include/hrbf/hrbf_core.h"
#include "include/hrbf/hrbf_phi_funcs.h"

#include <glm/glm.hpp>

typedef HRBF_fit<float, 3, Rbf_pow3<float> > DistanceField;

class FieldFunction
{
public:
    FieldFunction();
    ~FieldFunction();

    void Fit(const std::vector<DistanceField::Vector>& points,
             const std::vector<DistanceField::Vector>& normals,
             const float _r = 1.0f);
    void Fit(const std::vector<glm::vec3>& points,
             const std::vector<glm::vec3>& normals,
             const float _r = 1.0f);

    void SetR(const float _r);

    DistanceField::Scalar Eval(const DistanceField::Vector& x);
    float Eval(const glm::vec3& x);
    float EvalDist(const glm::vec3& x);

    DistanceField::Vector Grad(const DistanceField::Vector& x);
    glm::vec3 Grad(const glm::vec3& x);


private:
    /// @brief Method to remap distance field values to a compactly supported field function [0:1]
    /// @param float _distValue value we need to remap to be between [0:1]
    /// @return float field value between [0:1]
    float Remap(float _df);

    /// @brief Attribute used for remapping distance field to compactly supported field function.
    float r;


    /// @brief an HRBF distance field generator
    DistanceField m_distanceField;

};

#endif // FIELDFUNCTION_H
