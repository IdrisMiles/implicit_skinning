#ifndef COMPOSEDFIELD_H
#define COMPOSEDFIELD_H


#include <ScalarField/compositionop.h>
#include <ScalarField/fieldfunction.h>


class ComposedField
{
public:
    ComposedField();
    ~ComposedField();

    void SetFieldFuncA(std::shared_ptr<FieldFunction> _fieldFunctionA);

    void SetFieldFuncB(std::shared_ptr<FieldFunction> _fieldFunctionB);

    void SetFieldFunc(std::shared_ptr<FieldFunction> _fieldFunction,
                      int _id = 0);

    void SetFieldFunc(std::shared_ptr<FieldFunction> _fieldFunctionA,
                      std::shared_ptr<FieldFunction> _fieldFunctionB);

    void SetCompositionOp(std::shared_ptr<CompositionOp> _compositionOp);

    float Eval(glm::vec3 _x);

private:
    std::shared_ptr<CompositionOp> m_compositionOp;
    std::shared_ptr<FieldFunction> m_fieldFunctionA;
    std::shared_ptr<FieldFunction> m_fieldFunctionB;
};


#endif // COMPOSEDFIELD_H
