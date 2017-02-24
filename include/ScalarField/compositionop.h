#ifndef COMPOSITIONOP_H
#define COMPOSITIONOP_H

#include <memory>
#include <glm/glm.hpp>

#include "ScalarField/field1d.h"

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

    /// @brief Method to map angle to a value between [0:1]
    /// Refered to as controller dc(alpha) parameter for composition operator
    /// in "Robust Iso-Surface Tracking for Interactive Character Skinning"
    float Theta(const float _angleRadians);


private:

    Field1D m_field;

};

#endif // COMPOSITIONOP_H
