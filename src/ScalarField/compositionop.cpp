#include "ScalarField/compositionop.h"


CompositionOp::CompositionOp(const unsigned int _dim)
{
    m_field = Field1D(_dim, 0.0f);
}

CompositionOp::~CompositionOp()
{
}

void CompositionOp::Fit(const unsigned int _dim)
{
//    m_field.SetData(_dim, );
}

float CompositionOp::Eval(const float f1, const float f2, const float d)
{
    return f1 > f2 ? f1 : f2;
//    m_field.Eval(f1, f2, d);
}

float CompositionOp::Theta(const float _angleRadians)
{
    return _angleRadians;
}
