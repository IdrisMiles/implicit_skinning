#include "ScalarField/compositionop.h"


CompositionOp::CompositionOp(const unsigned int _dim):
    m_precomputed(false)
{
    // default theta function - pass through
    m_theta = [](float _angleRadians){
        return _angleRadians;
    };

    // default composition operator - Max
    m_compositionOp = [](float f1, float f2, float d){
        return f1 > f2 ? f1 : f2;
    };
}

CompositionOp::~CompositionOp()
{
}

void CompositionOp::SetTheta(std::function<float(float)> _theta)
{
    m_theta = _theta;
}

void CompositionOp::SetCompositionOp(std::function<float(float, float, float)> _compositionOp)
{
    m_compositionOp = _compositionOp;
}

void CompositionOp::Precompute(const unsigned int _dim)
{
    float data[_dim*_dim*_dim];

    for(unsigned int z=0; z<_dim; ++z)
    {
        for(unsigned int y=0; y<_dim; ++y)
        {
            for(unsigned int x=0; x<_dim; ++x)
            {
                float f1 = (float)x/_dim;
                float f2 = (float)y/_dim;
                float d = (float)z/_dim;

                float f = m_compositionOp(f1, f2, d);

                data[(z*_dim*_dim) + (y*_dim) + x] = f;
            }
        }
    }

    m_field.SetData(_dim, data);

    m_precomputed = true;
}

float CompositionOp::Eval(const float f1, const float f2, const float d)
{
    if(m_precomputed)
    {
        return m_field.Eval(f1, f2, d);
    }
    else
    {
        return m_compositionOp(f1, f2, d);
    }
}

float CompositionOp::Theta(const float _angleRadians)
{
    return m_theta(_angleRadians);
}


