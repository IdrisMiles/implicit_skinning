#include "ScalarField/fieldfunction.h"

FieldFunction::FieldFunction(glm::mat4 _transform) :
    m_transform(_transform)
{

}


FieldFunction::~FieldFunction()
{

}

void FieldFunction::Fit(const std::vector<glm::vec3>& points,
                        const std::vector<glm::vec3>& normals,
                        const float _r)
{
    r = _r;

    std::vector<DistanceField::Vector> DFVpoints;
    DFVpoints.reserve(points.size());
    std::vector<DistanceField::Vector> DFVnormals;
    DFVnormals.reserve(normals.size());

    for(auto &&p : points)
    {
        DFVpoints.emplace_back(DistanceField::Vector(p.x, p.y, p.z));
    }

    for(auto &&n : normals)
    {
        DFVnormals.emplace_back(DistanceField::Vector(n.x, n.y, n.z));
    }

    m_distanceField.hermite_fit(DFVpoints, DFVnormals);
}


void FieldFunction::SetR(const float _r)
{
    r = _r;
}


void FieldFunction::SetTransform(glm::mat4 _transform)
{
    m_transform = _transform;
}

float FieldFunction::Eval(const glm::vec3& x)
{
    glm::vec3 tx = TransformSpace(x);
    return Remap(m_distanceField.eval(DistanceField::Vector(tx.x, tx.y, tx.z)));
}

float FieldFunction::EvalDist(const glm::vec3& x)
{
    glm::vec3 tx = TransformSpace(x);
    return m_distanceField.eval(DistanceField::Vector(tx.x, tx.y, tx.z));
}

glm::vec3 FieldFunction::Grad(const glm::vec3& x)
{
    glm::vec3 tx = TransformSpace(x);
    auto g = m_distanceField.grad(DistanceField::Vector(tx.x, tx.y, tx.z));

    return glm::vec3(g(0), g(1), g(2));
}

float FieldFunction::Remap(float _df)
{
    float f = 0.0f;

    if(_df <= -r)
    {
        f = 1.0f;
    }
    else if(_df >= r)
    {

        f = 0.0f;
    }
    else
    {
        float x = _df / r;
        float x3 = x*x*x;
        float x5 = x*x*x*x*x;

        f = ((-3.0f * x5) / 16.0f) + ((5.0f * x3) / 8.0f) - ((15.0f * x) / 16.0f) + (0.5f);
    }

    return f;
}


glm::vec3 FieldFunction::TransformSpace(glm::vec3 _x)
{
    return glm::vec3(m_transform * glm::vec4(_x, 1.0f));
}
