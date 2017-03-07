#include "include/ScalarField/globalfieldfunction.h"
#include <algorithm>

GlobalFieldFunction::GlobalFieldFunction()
{

}

GlobalFieldFunction::~GlobalFieldFunction()
{
    m_compositionOps.clear();
    m_initialFields.clear();
    m_composedFields.clear();
}



float GlobalFieldFunction::Eval(const glm::vec3 &_x)
{
    std::vector<float> composedFieldValues(m_composedFields.size());
    int i=0;
    for(auto &cf : m_composedFields)
    {
        composedFieldValues[i++] = cf->Eval(_x);
    }

    return *std::max_element(composedFieldValues.begin(), composedFieldValues.end());
}

glm::vec3 GlobalFieldFunction::Grad(const glm::vec3 &_x)
{
    float h= 0.01f;
    float f = Eval(_x);

    float dx = (Eval(_x + glm::vec3(h, 0.0f, 0.0f)) - f) / h;
    float dy = (Eval(_x + glm::vec3(0.0f, h, 0.0f)) - f) / h;
    float dz = (Eval(_x + glm::vec3(0.0f, 0.0f, h)) - f) / h;

    return glm::vec3(dx, dy, dz);
}



void GlobalFieldFunction::AddCompositionOp(std::shared_ptr<CompositionOp> _compositionOp)
{
    m_compositionOps.push_back(_compositionOp);
    std::cout<<"comp op size:\t"<<m_compositionOps.size()<<"\n";
}

void GlobalFieldFunction::AddFieldFunction(std::shared_ptr<FieldFunction> _fieldFunction)
{
    m_initialFields.push_back(_fieldFunction);
    std::cout<<"Field func size:\t"<<m_initialFields.size()<<"\n";
}

void GlobalFieldFunction::AddComposedField(std::shared_ptr<ComposedField> _composedField)
{
    m_composedFields.push_back(_composedField);
    std::cout<<"comp fi size:\t"<<m_composedFields.size()<<"\n";
}
