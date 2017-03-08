#include "include/ScalarField/field1d.h"

Field1D::Field1D(unsigned int _dim, const float _initValue) :
    m_dim(_dim)
{
    m_data = new float[m_dim * m_dim * m_dim];

    for(unsigned int i=0; i<m_dim*m_dim*m_dim; ++i)
    {
        m_data[i] = _initValue;
    }

    m_textureSpaceTransform = [](glm::vec3 _v){return _v;};
}

Field1D::~Field1D()
{
    delete [] m_data;
}

void Field1D::operator=(const Field1D &_rhs)
{
    this->m_dim = _rhs.Dim();

    delete [] m_data;
    m_data = new float[m_dim * m_dim * m_dim];

    float *rhsData = _rhs.RawData();
    for(unsigned int i=0; i<m_dim*m_dim*m_dim; ++i)
    {
        m_data[i] = rhsData[i];
    }
}

unsigned int Field1D::Dim()const
{
    return m_dim;
}

float* Field1D::RawData()const
{
    return m_data;
}

void::Field1D::SetTextureSpaceTransform(std::function<glm::vec3 (glm::vec3)> _textureSpaceTransform)
{
    m_textureSpaceTransform = _textureSpaceTransform;
}

void Field1D::SetData(unsigned int _dim, float *_data)
{
    delete [] m_data;

    m_dim = _dim;
    m_data = new float[m_dim * m_dim * m_dim];

    for(unsigned int i=0; i<m_dim*m_dim*m_dim; ++i)
    {
        m_data[i] = _data[i];
    }
}

float Field1D::Eval(const glm::vec3 &_samplePoint)
{
    return TrilinearInterpolate(_samplePoint.x, _samplePoint.y, _samplePoint.z);
}

float Field1D::Eval(const float _x, const float _y, const float _z)
{
    return TrilinearInterpolate(_x, _y, _z);
}

float Field1D::TrilinearInterpolate(const float _x, const float _y, const float _z)
{

    //[0:1]
    glm::vec3 texSpace = m_textureSpaceTransform(glm::vec3(_x, _y, _z));

    // Get data coords
    float x = texSpace.x * m_dim;
    float y = texSpace.y * m_dim;
    float z = texSpace.z * m_dim;

    unsigned int x0 = floor(x);
    unsigned int y0 = floor(y);
    unsigned int z0 = floor(z);

    // Adjust for potential out of bounds
    x0 = x0 >= m_dim ? m_dim - 1 : x0;
    y0 = y0 >= m_dim ? m_dim - 1 : y0;
    z0 = z0 >= m_dim ? m_dim - 1 : z0;

    x0 = x0 < 0 ? 0 : x0;
    y0 = y0 < 0 ? 0 : y0;
    z0 = z0 < 0 ? 0 : z0;

    // Get other set of coords
    unsigned int x1 = x0 >= m_dim -1 ? x0 : x0 + 1;
    unsigned int y1 = y0 >= m_dim -1 ? y0 : y0 + 1;
    unsigned int z1 = z0 >= m_dim -1 ? z0 : z0 + 1;


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
    float dx = x - x0;
    float valX0 = LinearInterpolate(val0, val1, dx);
    float valX1 = LinearInterpolate(val2, val3, dx);
    float valX2 = LinearInterpolate(val4, val5, dx);
    float valX3 = LinearInterpolate(val6, val7, dx);

    float dy = y - y0;
    float valY0 = LinearInterpolate(valX0, valX1, dy);
    float valY1 = LinearInterpolate(valX2, valX3, dy);

    float dz = z - z0;
    float valZ0 = LinearInterpolate(valY0, valY1, dz);



    return valZ0;
}

float Field1D::LinearInterpolate(const float _f1, const float _f2, const float _t)
{
    float t = _t < 0.0f ? 0.0f : (_t > 1.0f ? 1.0f : _t);
    return _f1 + ((_f2-_f1) * _t);
}

int Field1D::Hash(const int _x, const int _y, const int _z)
{
    return _x + (_y * m_dim) + (_z * m_dim * m_dim);
}
