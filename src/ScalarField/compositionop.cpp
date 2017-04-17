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

//-----------------------------------------------------------------------------------------------------

CompositionOp::~CompositionOp()
{
}

//-----------------------------------------------------------------------------------------------------

void CompositionOp::SetTheta(std::function<float(float)> _theta)
{
    m_theta = _theta;
}

//-----------------------------------------------------------------------------------------------------

void CompositionOp::SetCompositionOp(std::function<float(float, float, float)> _compositionOp)
{
    m_compositionOp = _compositionOp;
}

//-----------------------------------------------------------------------------------------------------

void CompositionOp::SetParams(float _alpha0, float _alpha1, float _alpha2,
                              float _theta0, float _theta1, float _theta2,
                              float _w0, float _w1)
{
    m_alpha0 = _alpha0;
    m_alpha1 = _alpha1;
    m_alpha2 = _alpha2;

    m_theta0 = _theta0;
    m_theta1 = _theta1;
    m_theta2 = _theta2;

    m_w0 = _w0;
    m_w1 = _w1;

    // Set Theta - the opening function
    m_theta = [this](float alpha)->float{

        auto k = [](float x)->float{
            1.0f - exp(1.0f - (1.0f / (1.0f - exp(1.0f - (1.0f/x)))));
        };


        if(alpha <= m_alpha0)
        {
            return m_theta0;
        }
        else if(alpha >= m_alpha2)
        {
            return m_theta2;
        }
        else if(alpha > m_alpha0 && alpha <= m_alpha1)
        {
            float a = pow(k((alpha-m_alpha1)/(m_alpha0-m_alpha1)), m_w0);
            return (a*(m_theta0 - m_theta1)) + m_theta1;
        }
        else if(alpha > m_alpha1 && alpha < m_alpha2)
        {
            float a = pow(k((alpha-m_alpha1)/(m_alpha2-m_alpha1)), m_w1);
            return (a*(m_theta2 - m_theta1)) + m_theta1;
        }
        else
        {
            assert(false);
            return 0.0f;
        }
    };

    m_kTheta = [this](float f)->float{
        if(f >= 0.5f)
        {
        }
        else
        {
        }

    };

    m_gHat = [this](float f1, float f2)->float{

    };


    m_compositionOp = [this](float f1, float f2, float d)->float{

    };
}

//-----------------------------------------------------------------------------------------------------

void CompositionOp::Precompute(const unsigned int _res)
{
    float data[_res*_res*_res];
    float4 *cuGrad = new float4[_res*_res*_res];

    // field value
    for(unsigned int z=0; z<_res; ++z)
    {
        for(unsigned int y=0; y<_res; ++y)
        {
            for(unsigned int x=0; x<_res; ++x)
            {
                float f1 = (float)x/_res;
                float f2 = (float)y/_res;
                float d = (float)z/_res;
                float Ci = f1;


                // old
                float f = m_compositionOp(f1, f2, d);
                data[(z*_res*_res) + (y*_res) + x] = f;
            }
        }
    }

    // gradient
    for(unsigned int z=0; z<_res; ++z)
    {
        for(unsigned int y=0; y<_res; ++y)
        {
            for(unsigned int x=0; x<_res; ++x)
            {
                int id = (z*_res*_res) + (y*_res) + x;
                int right = (z*_res*_res) + (y*_res) + (x+1);
                int left = (z*_res*_res) + (y*_res) + (x-1);
                int up = (z*_res*_res) + ((y+1)*_res) + (x);
                int down = (z*_res*_res) + ((y-1)*_res) + (x);
                int forward = ((z+1)*_res*_res) + (y*_res) + (x);
                int backward = ((z-1)*_res*_res) + (y*_res) + (x);

                float x1, x2, y1, y2, z1, z2;

                x1 = (x==0) ? data[right] : data[id];
                x2 = (x==0) ? data[id] : data[left];

                y1 = (y==0) ? data[up] : data[id];
                y2 = (y==0) ? data[id] : data[down];

                z1 = (z==0) ? data[forward] : data[id];
                z2 = (z==0) ? data[id] : data[backward];


                cuGrad[id] = make_float4(x1-x2, y1-y2, z1-z2, data[id]);

            }
        }
    }



    m_field.SetData(_res, data);
    d_field.CreateCudaTexture(_res, data, cudaFilterModeLinear);
    d_grad.CreateCudaTexture(_res, cuGrad, cudaFilterModeLinear);
    delete [] cuGrad;

    m_precomputed = true;
}

//-----------------------------------------------------------------------------------------------------

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

//-----------------------------------------------------------------------------------------------------

float CompositionOp::Theta(const float _angleRadians)
{
    return m_theta(_angleRadians);
}


//-----------------------------------------------------------------------------------------------------

cudaTextureObject_t &CompositionOp::GetFieldFunc3DTexture()
{
    return d_field.GetCudaTextureObject();
}

//-----------------------------------------------------------------------------------------------------

cudaTextureObject_t &CompositionOp::GetFieldGrad3DTexture()
{
    return d_grad.GetCudaTextureObject();
}

//-----------------------------------------------------------------------------------------------------
