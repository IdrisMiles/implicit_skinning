#ifndef FIELD1D_H
#define FIELD1D_H

#include <functional>

#include <glm/glm.hpp>

class Field1D
{
public:
    Field1D(unsigned int _dim = 32, const float _initValue = 0.0f);
    ~Field1D();
    void operator=(const Field1D &_rhs);

    unsigned int Dim() const;
    float* RawData()const;

    void SetTextureSpaceTransform(std::function<glm::vec3(glm::vec3)> _textureSpaceTransform);
    void SetData(unsigned int _dim, float *_data);
    float Eval(const glm::vec3 &_samplePoint);
    float Eval(const float _x, const float _y, const float _z);

private:
    float TrilinearInterpolate(const float _x, const float _y, const float _z);
    float LinearInterpolate(const float _f1, const float _f2, const float _t);
    int Hash(const int _x, const int _y, const int _z);

    /// @brief This attribute transform a local/word space coord into texture space
    /// so that it can be used to sample the 3D texture/data.
    /// This could be a matrix, but I wanted to be funky, plus this allows for complex transforms
    std::function<glm::vec3(glm::vec3)> m_textureSpaceTransform;

    unsigned int m_dim;
    float *m_data;
};

#endif // FIELD1D_H
