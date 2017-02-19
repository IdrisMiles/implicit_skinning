#ifndef COMPOSITIONOP_H
#define COMPOSITIONOP_H

#include <memory>
#include <glm/glm.hpp>

class CompositionOp
{
public:
    CompositionOp(const unsigned int _dim = 32);
    ~CompositionOp();

    /// @brief Method to initialise the 3D array holding the evaluations
    /// of the composition operator
    void Fit(const unsigned int _dim = 32);

    /// @brief Method to compute the result value of the composed field functions
    float Eval(const float f1, const float f2, const float d);

    /// @brief Method to compute the gradient of the composed field functions
    //glm::vec3 Grad(const float f1, const float f2, const float d);

    /// @brief Method to map angle to a value between [0:1]
    /// Refered to as controller dc(alpha) parameter for composition operator
    /// in "Robust Iso-Surface Tracking for Interactive Character Skinning"
    float Theta(const float _angleRadians);


private:
    float TrilinearInterpolate(const float f1, const float f2, const float d);

    float LinearInterpolate(const float f1, const float f2, const float d);

    int Hash(const int x, const int y, const int z);

    unsigned int m_dim;
    float *m_data;
};

#endif // COMPOSITIONOP_H
