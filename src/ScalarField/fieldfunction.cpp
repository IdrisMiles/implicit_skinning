#include "ScalarField/fieldfunction.h"
#include <glm/gtx/string_cast.hpp>
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

void FieldFunction::PrecomputeField()
{
    unsigned int dim = 32;
    float data[dim*dim*dim] = {0.0f};
    float scale = 800.0f;

    for(unsigned int z=0; z<dim; ++z)
    {
        for(unsigned int y=0; y<dim; ++y)
        {
            for(unsigned int x=0; x<dim; ++x)
            {
                glm::vec3 point(scale*((((float)x/dim)*2.0f)-1.0f),
                                scale*((((float)y/dim)*2.0f)-1.0f),
                                scale*((((float)z/dim)*2.0f)-1.0f));

                glm::vec3 tx = TransformSpace(point);
                float d = Remap(m_distanceField.eval(DistanceField::Vector(tx.x, tx.y, tx.z)));

                data[z*dim*dim + y*dim+ x] = d;
            }
        }
    }

    m_field.SetData(dim, data);
    m_field.SetTextureSpaceTransform([](glm::vec3 x){
        float scale = 1.0f/7.5f;
        return (((x*scale)+glm::vec3(1.0f,1.0f,1.0f))*0.5f);
    });

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
    return m_field.Eval(tx);


//    return Remap(m_distanceField.eval(DistanceField::Vector(tx.x, tx.y, tx.z)));
}

float FieldFunction::EvalDist(const glm::vec3& x)
{
    glm::vec3 tx = TransformSpace(x);
    return m_distanceField.eval(DistanceField::Vector(tx.x, tx.y, tx.z));
}

glm::vec3 FieldFunction::Grad(const glm::vec3& x)
{
    return glm::vec3(0.0f, 1.0f, 0.0f);


//    glm::vec3 tx = TransformSpace(x);
//    const static float scale = 1.0f/7.5f;
//    float h= 0.01f;
//    float h2 = 2.0f*h;
//    glm::vec3 nx = (((tx*scale)+glm::vec3(1.0f,1.0f,1.0f))*0.5f);
//    float f = m_field.Eval(nx);
//    float dx = (m_field.Eval(nx + glm::vec3(h, 0.0f, 0.0f)) - f) / h2;
//    float dy = (m_field.Eval(nx + glm::vec3(0.0f, h, 0.0f)) - f) / h2;
//    float dz = (m_field.Eval(nx + glm::vec3(0.0f, 0.0f, h)) - f) / h2;

//    return glm::vec3(dx, dy, dz);



//    glm::vec3 tx = TransformSpace(x);
//    auto g = m_distanceField.grad(DistanceField::Vector(tx.x, tx.y, tx.z));

//    return glm::vec3(g(0), g(1), g(2));
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
