#include "include/ScalarField/globalfieldfunction.h"
#include <algorithm>

GlobalFieldFunction::GlobalFieldFunction()
{

}

GlobalFieldFunction::~GlobalFieldFunction()
{
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

void GlobalFieldFunction::AddComposedField(std::shared_ptr<ComposedField> _composedField)
{
    m_composedFields.push_back(_composedField);
}
