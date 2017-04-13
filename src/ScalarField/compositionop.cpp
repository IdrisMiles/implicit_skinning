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

void CompositionOp::Precompute(const unsigned int _res)
{
    float data[_res*_res*_res];
//    float4 *cuGrad = new float4[_dim*_dim*_dim];

    for(unsigned int z=0; z<_res; ++z)
    {
        for(unsigned int y=0; y<_res; ++y)
        {
            for(unsigned int x=0; x<_res; ++x)
            {
                float f1 = (float)x/_res;
                float f2 = (float)y/_res;
                float d = (float)z/_res;

                float f = m_compositionOp(f1, f2, d);

                data[(z*_res*_res) + (y*_res) + x] = f;
            }
        }
    }

    m_field.SetData(_res, data);
    d_field.CreateCudaTexture(_res, data, cudaFilterModeLinear);
//    d_grad.CreateCudaTexture(_dim, cuGrad, cudaFilterModeLinear);
//    delete [] cuGrad;

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


cudaTextureObject_t &CompositionOp::GetFieldFunc3DTexture()
{
    return d_field.GetCudaTextureObject();
}
