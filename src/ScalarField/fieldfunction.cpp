#include "ScalarField/fieldfunction.h"
#include <glm/gtx/string_cast.hpp>
#include <float.h>


FieldFunction::FieldFunction(glm::mat4 _transform) :
    m_transform(_transform),
    m_fit(false)
{
    m_cachedEvals.reserve(4);
}


FieldFunction::~FieldFunction()
{

}

void FieldFunction::Fit(const std::vector<glm::vec3>& points,
                        const std::vector<glm::vec3>& normals,
                        const float _r)
{
    if(points.size() < 1 || normals.size() < 1)
    {
        m_fit = false;
        return;
    }

    m_r = _r;

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

    m_fit = true;
}

void FieldFunction::PrecomputeField(const unsigned int _dim, const float _scale)
{
    if(!m_fit)
    {
        return;
    }

    float data[_dim*_dim*_dim] = {0.0f};

    for(unsigned int z=0; z<_dim; ++z)
    {
        for(unsigned int y=0; y<_dim; ++y)
        {
            for(unsigned int x=0; x<_dim; ++x)
            {
                glm::vec3 point(_scale*((((float)x/_dim)*2.0f)-1.0f),
                                _scale*((((float)y/_dim)*2.0f)-1.0f),
                                _scale*((((float)z/_dim)*2.0f)-1.0f));

                glm::vec3 tx = TransformSpace(point);
                float d = Remap(m_distanceField.eval(DistanceField::Vector(tx.x, tx.y, tx.z)));

                data[z*_dim*_dim + y*_dim+ x] = d;
            }
        }
    }

    m_field.SetData(_dim, data);
    m_field.SetTextureSpaceTransform([_scale](glm::vec3 x){
        return (((x/_scale)+glm::vec3(1.0f,1.0f,1.0f))*0.5f);
    });

}

void FieldFunction::SetR(const float _r)
{
    m_r = _r;
}

void FieldFunction::SetTransform(glm::mat4 _transform)
{
    m_transform = _transform;
}

bool FieldFunction::Equiv(glm::vec3 _a, glm::vec3 _b)
{
    glm::bvec3 result = glm::equal(_a, _b);
    return result.x && result.b && result.z;
}

float FieldFunction::Eval(const glm::vec3 &_x)
{
    if(!m_fit)
    {
        return 0.0f;
    }

    // checked cached results first
    for(auto &ce : m_cachedEvals)
    {
        if(Equiv(_x, ce.first))
        {
            return ce.second;
        }
    }


    // need to evaluate
    glm::vec3 tx = TransformSpace(_x);
    float f = m_field.Eval(tx);
//    float f = Remap(m_distanceField.eval(DistanceField::Vector(tx.x, tx.y, tx.z)));

    // cache value
    if(m_cachedEvals.size() >= 4)
    {
        m_cachedEvals.erase(m_cachedEvals.begin());
    }
    m_cachedEvals.push_back(std::make_pair(_x, f));


    return f;
}

float FieldFunction::EvalDist(const glm::vec3& x)
{
    if(!m_fit)
    {
        return 0.0f;
    }

    glm::vec3 tx = TransformSpace(x);
    return m_distanceField.eval(DistanceField::Vector(tx.x, tx.y, tx.z));
}

glm::vec3 FieldFunction::Grad(const glm::vec3& x)
{
    if(!m_fit)
    {
        return glm::vec3(0.0f, 1.0f, 0.0f);
    }

    // check cached values
    for(auto &cg : m_cachedGrads)
    {
        if(Equiv(x, cg.first))
        {
            return cg.second;
        }
    }


    // forward difference gradient
//    glm::vec3 tx = TransformSpace(x);
//    float h= 0.01f;
//    float f = m_field.Eval(tx);

//    float dx = (m_field.Eval(tx + glm::vec3(h, 0.0f, 0.0f)) - f) / h;
//    float dy = (m_field.Eval(tx + glm::vec3(0.0f, h, 0.0f)) - f) / h;
//    float dz = (m_field.Eval(tx + glm::vec3(0.0f, 0.0f, h)) - f) / h;

//    glm::vec3 grad(dx, dy, dz);
//    return grad;


    // more accurate but heavy gradient computation
    glm::vec3 tx = TransformSpace(x);
    auto g = m_distanceField.grad(DistanceField::Vector(tx.x, tx.y, tx.z));
    glm::vec3 grad(g(0), g(1), g(2));


//    // cache value
    if(m_cachedGrads.size() >= 4)
    {
        m_cachedGrads.erase(m_cachedGrads.begin());
    }
    m_cachedGrads.push_back(std::make_pair(x, grad));


    return grad;
}

float FieldFunction::Remap(float _df)
{
    float f = 0.0f;

    if(_df <= -m_r)
    {
        f = 1.0f;
    }
    else if(_df >= m_r)
    {

        f = 0.0f;
    }
    else
    {
        float x = _df / m_r;
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
