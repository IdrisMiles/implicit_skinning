#ifndef COMPOSITIONOP_H
#define COMPOSITIONOP_H

#include <memory>
#include <glm/glm.hpp>

class CompositionOp
{
public:
    CompositionOp(const unsigned int _dim = 32);
    ~CompositionOp();

    void Fit(const unsigned int _dim = 32);
    float Eval(const float f1, const float f2, const float d);
    glm::vec3 Grad(const float f1, const float f2, const float d);


private:
    float TrilinearInterpolate(const float f1, const float f2, const float d);
    float LinearInterpolate(const float f1, const float f2, const float d);
    int Hash(const int x, const int y, const int z);

    unsigned int m_dim;
    std::shared_ptr<float> m_data;
};

#endif // COMPOSITIONOP_H
