#ifndef COMPOSEDFIELD_H
#define COMPOSEDFIELD_H

//-------------------------------------------------------------------------------

#include <memory>

#include <ScalarField/compositionop.h>
#include <ScalarField/fieldfunction.h>


//-------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 18/04/2017
//-------------------------------------------------------------------------------


/// @class ComposeField
/// @brief CPU implementation of composed field
class ComposedField
{
public:
    /// @brief constructor
    ComposedField();

    /// @brief destructor
    ~ComposedField();


    /// @brief method to set field A used in composed field
    void SetFieldFuncA(std::shared_ptr<FieldFunction> _fieldFunctionA);

    /// @brief method to set field B used in composed field
    void SetFieldFuncB(std::shared_ptr<FieldFunction> _fieldFunctionB);

    /// @brief method to set field A or B used in composed field
    void SetFieldFunc(std::shared_ptr<FieldFunction> _fieldFunction,
                      int _id = 0);

    /// @brief method to set field A & B used in composed field
    void SetFieldFunc(std::shared_ptr<FieldFunction> _fieldFunctionA,
                      std::shared_ptr<FieldFunction> _fieldFunctionB);

    /// @brief method to set composition operator used to compose field
    /// @param _compositionOp
    void SetCompositionOp(std::shared_ptr<CompositionOp> _compositionOp);

    /// @brief method to evaluate composed field at sample point on CPU
    /// @param _x : sample point
    float Eval(glm::vec3 _x);

private:
    /// @brief composition operator
    std::shared_ptr<CompositionOp> m_compositionOp;

    /// @brief field A
    std::shared_ptr<FieldFunction> m_fieldFunctionA;

    /// @brief field B
    std::shared_ptr<FieldFunction> m_fieldFunctionB;
};

//-------------------------------------------------------------------------------

#endif // COMPOSEDFIELD_H
