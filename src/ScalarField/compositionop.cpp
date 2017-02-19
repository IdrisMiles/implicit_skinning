#include "ScalarField/compositionop.h"


CompositionOp::CompositionOp(const unsigned int _dim)
{
    m_dim = _dim;
    m_data = new float[m_dim*m_dim*m_dim];
}

CompositionOp::~CompositionOp()
{
    delete m_data;
}

void CompositionOp::Fit(const unsigned int _dim)
{
    m_dim = _dim;
    delete m_data;
    m_data = new float[m_dim*m_dim*m_dim];
}

float CompositionOp::Eval(const float f1, const float f2, const float d)
{
    return f1 > f2 ? f1 : f2;
//    return TrilinearInterpolate(f1, f2, d);
}

//glm::vec3 CompositionOp::Grad(const float f1, const float f2, const float d)
//{
//    return glm::vec3(0.0f, 1.0f, 0.0f);
////    float h = 0.01;

////    float x = (TrilinearInterpolate(f1-h, f2, d) - TrilinearInterpolate(f1+h, f2, d)) / 2.0f*h;
////    float y = (TrilinearInterpolate(f1, f2-h, d) - TrilinearInterpolate(f1, f2+h, d)) / 2.0f*h;
////    float z = (TrilinearInterpolate(f1, f2, d-h) - TrilinearInterpolate(f1, f2, d+h)) / 2.0f*h;

////    return glm::normalize(glm::vec3(x, y, z));
//}

float CompositionOp::Theta(const float _angleRadians)
{
    return _angleRadians;
}

float CompositionOp::TrilinearInterpolate(const float f1, const float f2, const float d)
{
    // Get data coords
    unsigned int x0 = floor(f1 * m_dim);
    unsigned int y0 = floor(f2 * m_dim);
    unsigned int z0 = floor(d * m_dim);

    // Adjust for potential out of bounds
    x0 = x0 >= m_dim ? m_dim - 1 : x0;
    y0 = y0 >= m_dim ? m_dim - 1 : y0;
    z0 = z0 >= m_dim ? m_dim - 1 : z0;

    x0 = x0 < 0 ? 0 : x0;
    y0 = y0 < 0 ? 0 : y0;
    z0 = z0 < 0 ? 0 : z0;

    // Get other set of coords
    unsigned int x1 = x0 == m_dim -1 ? x0 : x0 + 1;
    unsigned int y1 = y0 == m_dim -1 ? y0 : y0 + 1;
    unsigned int z1 = z0 == m_dim -1 ? z0 : z0 + 1;


    // Get values
    float val0 = m_data[Hash(x0, y0, z0)]; //bottom font left
    float val1 = m_data[Hash(x1, y0, z0)]; //bottom front right
    float val2 = m_data[Hash(x0, y1, z0)]; //bottom back left
    float val3 = m_data[Hash(x1, y1, z0)]; //bottom back right

    float val4 = m_data[Hash(x0, y0, z1)];
    float val5 = m_data[Hash(x1, y0, z1)];
    float val6 = m_data[Hash(x0, y1, z1)];
    float val7 = m_data[Hash(x1, y1, z1)];


    // do interpolation here
    float valX0 = LinearInterpolate(val0, val1, f1);
    float valX1 = LinearInterpolate(val2, val3, f1);
    float valX2 = LinearInterpolate(val4, val5, f1);
    float valX3 = LinearInterpolate(val6, val7, f1);

    float valY0 = LinearInterpolate(valX0, valX2, f2);
    float valY1 = LinearInterpolate(valX1, valX3, f2);

    float valZ0 = LinearInterpolate(valY0, valY1, d);



    return valZ0;
}

float CompositionOp::LinearInterpolate(const float f1, const float f2, const float d)
{
    return f1 + ((f2-f1) * d);
}

int CompositionOp::Hash(const int x, const int y, const int z)
{
    return x + (y * m_dim) + (z * m_dim * m_dim);
}
