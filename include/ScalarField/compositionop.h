#ifndef COMPOSITIONOP_H
#define COMPOSITIONOP_H

#include <functional>

#include <glm/glm.hpp>

#include "ScalarField/field1d.h"

class CompositionOp
{
public:
    CompositionOp(const unsigned int _dim = 32);
    ~CompositionOp();

    void SetTheta(std::function<float(float)> _theta);
    void SetCompositionOp(std::function<float(float, float, float)> _compositionOp);
    void Precompute(const unsigned int _dim = 32);

    /// @brief Method to compute the result value of the composed field functions
    float Eval(const float f1, const float f2, const float d);

    /// @brief Method to map angle to a value between [0:1]
    /// Refered to as controller dc(alpha) parameter for composition operator
    /// in "Robust Iso-Surface Tracking for Interactive Character Skinning"
    float Theta(const float _angleRadians);


private:
    std::function<float(float)> m_theta;

    std::function<float(float, float, float)> m_compositionOp;


    Field1D<float> m_field;
    bool m_precomputed;

};

#endif // COMPOSITIONOP_H
