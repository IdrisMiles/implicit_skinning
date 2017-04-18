#include "ScalarField/fieldfunction.h"

#include <float.h>

#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>

FieldFunction::FieldFunction() :
    m_fit(false),
    m_precomputedGPU(false),
    m_precomputedCPU(false),
    m_transform(glm::mat4(1.0f)),
    m_textureSpaceTransform(glm::mat4(1.0f))
{
}

//------------------------------------------------------------------------------------------------

FieldFunction::~FieldFunction()
{

}

//------------------------------------------------------------------------------------------------

void FieldFunction::Fit(const std::vector<glm::vec3>& points,
                        const std::vector<glm::vec3>& normals,
                        const float _r)
{
    if(points.size() < 3 || normals.size() < 3)
    {
        m_fit = false;
        return;
    }

    m_supportRad = _r;

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

//------------------------------------------------------------------------------------------------

void FieldFunction::PrecomputeField(const unsigned int _res, const float _dim)
{
    float data[_res*_res*_res];
    glm::vec3 grad[_res*_res*_res];
    float4 *cuGrad = new float4[_res*_res*_res];

    for(unsigned int z=0; z<_res; ++z)
    {
        for(unsigned int y=0; y<_res; ++y)
        {
            for(unsigned int x=0; x<_res; ++x)
            {
                glm::vec3 point(_dim*((((float)x/_res)*2.0f)-1.0f),
                                _dim*((((float)y/_res)*2.0f)-1.0f),
                                _dim*((((float)z/_res)*2.0f)-1.0f));

                glm::vec3 tx = TransformSpace(point);
                auto samplePoint = DistanceField::Vector(tx.x, tx.y, tx.z);

                float d = 0.0f;
                DistanceField::Vector g(0.0f, 0.0f, 0.0f);
                if(m_fit)
                {
                    d = Remap(m_distanceField.eval(samplePoint));
                    g = m_distanceField.grad(samplePoint);
                }

                data[(z*_res*_res) + (y*_res)+ x] = d;
                grad[(z*_res*_res) + (y*_res)+ x] = glm::vec3(g(0), g(1), g(2));
                cuGrad[(z*_res*_res) + (y*_res)+ x] = make_float4(g(0), g(1), g(2), d);
            }
        }
    }

    if(m_fit)
    {
        m_field.SetData(_res, data);
        m_grad.SetData(_res, grad);
        m_precomputedCPU = true;
    }
    d_field.CreateCudaTexture(_res, data, cudaFilterModeLinear);
    d_grad.CreateCudaTexture(_res, cuGrad, cudaFilterModeLinear);
    delete [] cuGrad;
    m_precomputedGPU = true;


    // create texture space transform
    m_textureSpaceTransform = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 0.5f, 0.5f));
    m_textureSpaceTransform = glm::translate(m_textureSpaceTransform, glm::vec3(1.0f, 1.0f, 1.0f));
    m_textureSpaceTransform = glm::scale(m_textureSpaceTransform, glm::vec3(1.0f/_dim, 1.0f/_dim, 1.0f/_dim));


    m_field.SetTextureSpaceTransform(m_textureSpaceTransform);
    m_grad.SetTextureSpaceTransform(m_textureSpaceTransform);

}

//------------------------------------------------------------------------------------------------

void FieldFunction::SetSupportRadius(const float _r)
{
    m_supportRad = _r;
}

//------------------------------------------------------------------------------------------------

void FieldFunction::SetTransform(glm::mat4 _transform)
{
    m_transform = _transform;
}

//------------------------------------------------------------------------------------------------

glm::mat4 FieldFunction::GetTransform() const
{
    return m_transform;
}

//------------------------------------------------------------------------------------------------

glm::mat4 FieldFunction::GetTextureSpaceTransform() const
{
    return m_textureSpaceTransform;
}

//------------------------------------------------------------------------------------------------

float FieldFunction::Eval(const glm::vec3 &_x)
{
    if(!m_fit)
    {
        return 0.0f;
    }

    glm::vec3 tx = TransformSpace(_x);

    // texture lookup
    float f = m_field.Eval(tx);

    // accurate evaluation
//    float f = Remap(m_distanceField.eval(DistanceField::Vector(tx.x, tx.y, tx.z)));


    return f;
}

//------------------------------------------------------------------------------------------------

float FieldFunction::EvalDist(const glm::vec3& x)
{
    if(!m_fit)
    {
        return 0.0f;
    }

    glm::vec3 tx = TransformSpace(x);
    return m_distanceField.eval(DistanceField::Vector(tx.x, tx.y, tx.z));
}

//------------------------------------------------------------------------------------------------

glm::vec3 FieldFunction::Grad(const glm::vec3& x)
{
    if(!m_fit)
    {
        return glm::vec3(0.0f, 1.0f, 0.0f);
    }

    glm::vec3 tx = TransformSpace(x);


    // texture lookup
    glm::vec3 grad = m_grad.Eval(tx);

    // more accurate but heavy gradient computation
//    glm::vec3 tx = TransformSpace(x);
//    auto g = m_distanceField.grad(DistanceField::Vector(tx.x, tx.y, tx.z));
//    glm::vec3 grad(g(0), g(1), g(2));


    return grad;
}

//------------------------------------------------------------------------------------------------

cudaTextureObject_t &FieldFunction::GetFieldFuncCudaTextureObject()
{
    return d_field.GetCudaTextureObject();
}

//------------------------------------------------------------------------------------------------

cudaTextureObject_t &FieldFunction::GetFieldGradCudaTextureObject()
{
    return d_grad.GetCudaTextureObject();
}

//------------------------------------------------------------------------------------------------

float FieldFunction::Remap(float _df)
{
    float f = 0.0f;

    if(_df <= -m_supportRad)
    {
        f = 1.0f;
    }
    else if(_df >= m_supportRad)
    {

        f = 0.0f;
    }
    else
    {
        float x = _df / m_supportRad;
        float x3 = x*x*x;
        float x5 = x*x*x*x*x;

        f = ((-3.0f * x5) / 16.0f) + ((5.0f * x3) / 8.0f) - ((15.0f * x) / 16.0f) + (0.5f);
    }

    return f;
}

//------------------------------------------------------------------------------------------------

glm::vec3 FieldFunction::TransformSpace(glm::vec3 _x)
{
    return glm::vec3(m_transform * glm::vec4(_x, 1.0f));
}

//------------------------------------------------------------------------------------------------
