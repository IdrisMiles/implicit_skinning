#ifndef FIELD1D_H
#define FIELD1D_H

#include <functional>

#include <glm/glm.hpp>

template <typename T>
class Field1D
{
public:
    Field1D(unsigned int _dim = 32)
    {
        m_dim = _dim;
        m_data = new T[m_dim * m_dim * m_dim];


        m_textureSpaceTransform = [](glm::vec3 _v){return _v;};
    }

    ~Field1D()
    {
        if(m_data != nullptr)
        {
            delete [] m_data;
            m_data = nullptr;
        }
    }

    unsigned int Dim() const
    {
        return m_dim;
    }

    T* RawData()const
    {
        return m_data;
    }

    void SetTextureSpaceTransform(std::function<glm::vec3(glm::vec3)> _textureSpaceTransform)
    {
        m_textureSpaceTransform = _textureSpaceTransform;
    }

    void SetData(unsigned int _dim, T *_data)
    {
        if(m_data != nullptr)
        {
            delete [] m_data;
            m_data = nullptr;
        }

        m_dim = _dim;
        m_data = new T[m_dim * m_dim * m_dim];

        for(unsigned int i=0; i<m_dim*m_dim*m_dim; ++i)
        {
            m_data[i] = _data[i];
        }
    }

    T Eval(const glm::vec3 &_samplePoint)
    {
        return TrilinearInterpolate(_samplePoint.x, _samplePoint.y, _samplePoint.z);
    }

    T Eval(const float _x, const float _y, const float _z)
    {
        return TrilinearInterpolate(_x, _y, _z);
    }

private:
    T TrilinearInterpolate(const float _x, const float _y, const float _z)
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
        T val0 = m_data[Hash(x0, y0, z0)]; //bottom font left
        T val1 = m_data[Hash(x1, y0, z0)]; //bottom front right
        T val2 = m_data[Hash(x0, y1, z0)]; //bottom back left
        T val3 = m_data[Hash(x1, y1, z0)]; //bottom back right

        T val4 = m_data[Hash(x0, y0, z1)];
        T val5 = m_data[Hash(x1, y0, z1)];
        T val6 = m_data[Hash(x0, y1, z1)];
        T val7 = m_data[Hash(x1, y1, z1)];


        // do interpolation here
        float dx = x - x0;
        T valX0 = LinearInterpolate(val0, val1, dx);
        T valX1 = LinearInterpolate(val2, val3, dx);
        T valX2 = LinearInterpolate(val4, val5, dx);
        T valX3 = LinearInterpolate(val6, val7, dx);

        float dy = y - y0;
        T valY0 = LinearInterpolate(valX0, valX1, dy);
        T valY1 = LinearInterpolate(valX2, valX3, dy);

        float dz = z - z0;
        T valZ0 = LinearInterpolate(valY0, valY1, dz);



        return valZ0;
    }

    T LinearInterpolate(const T _f1, const T _f2, const float _t)
    {
        float t = _t < 0.0f ? 0.0f : (_t > 1.0f ? 1.0f : _t);
        return _f1 + ((_f2-_f1) * t);
    }

    int Hash(const int _x, const int _y, const int _z)
    {
        return _x + (_y * m_dim) + (_z * m_dim * m_dim);
    }


    /// @brief This attribute transform a local/word space coord into texture space
    /// so that it can be used to sample the 3D texture/data.
    /// This could be a matrix, but I wanted to be funky, plus this allows for complex transforms
    std::function<glm::vec3(glm::vec3)> m_textureSpaceTransform;

    unsigned int m_dim;
    T *m_data;
};



#endif // FIELD1D_H
