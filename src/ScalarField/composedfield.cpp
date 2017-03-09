#include "include/ScalarField/composedfield.h"
#include <glm/gtx/vector_angle.hpp>

ComposedField::ComposedField()
{

}

ComposedField::~ComposedField()
{
    m_compositionOp = nullptr;
    m_fieldFunctionA = nullptr;
    m_fieldFunctionB = nullptr;
}

void ComposedField::SetFieldFuncA(std::shared_ptr<FieldFunction> _fieldFunctionA)
{
    m_fieldFunctionA = _fieldFunctionA;
}

void ComposedField::SetFieldFuncB(std::shared_ptr<FieldFunction> _fieldFunctionB)
{
    m_fieldFunctionB = _fieldFunctionB;
}

void ComposedField::SetFieldFunc(std::shared_ptr<FieldFunction> _fieldFunction,
                                 int _id)
{
    if(_id == 0)
    {
        m_fieldFunctionA = _fieldFunction;
    }
    else if(_id == 1)
    {
        m_fieldFunctionB = _fieldFunction;
    }
}

void ComposedField::SetFieldFunc(std::shared_ptr<FieldFunction> _fieldFunctionA,
                                 std::shared_ptr<FieldFunction> _fieldFunctionB)
{
    m_fieldFunctionA = _fieldFunctionA;
    m_fieldFunctionB = _fieldFunctionB;
}


void ComposedField::SetCompositionOp(std::shared_ptr<CompositionOp> _compositionOp)
{
    m_compositionOp = _compositionOp;
}


float ComposedField::Eval(glm::vec3 _x)
{
    // Do a bit of error checking
    if(m_compositionOp == nullptr)
    {
        std::cout<<"Composition op null in interiornode\n";
    }


    // Do actual evaluation
    float f1 = 0.0f;
    float f2 = 0.0f;
    float d = 0.0f;
    if(m_fieldFunctionA != nullptr)
    {
        f1 = m_fieldFunctionA->Eval(_x);
    }

    if(m_fieldFunctionB != nullptr)
    {
        f2 = m_fieldFunctionB->Eval(_x);
    }

    if(m_fieldFunctionA != nullptr && m_fieldFunctionB != nullptr)
    {
        glm::vec3 g1 = m_fieldFunctionA->Grad(_x);
        glm::vec3 g2 = m_fieldFunctionB->Grad(_x);
        float angle = glm::angle(g1, g2);

        d = m_compositionOp->Theta(angle);
    }

    return m_compositionOp->Eval(f1, f2, d);
}
