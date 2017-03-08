#ifndef GLOBALFIELDFUNCTION_H
#define GLOBALFIELDFUNCTION_H

#include <ScalarField/compositionop.h>
#include <ScalarField/fieldfunction.h>
#include <ScalarField/composedfield.h>

#include <memory>
#include <vector>


class GlobalFieldFunction
{
public:
    GlobalFieldFunction();
    ~GlobalFieldFunction();


    float Eval(const glm::vec3 &_x);

    glm::vec3 Grad(const glm::vec3 &_x);


    void AddComposedField(std::shared_ptr<ComposedField> _composedField);

private:

    std::vector<std::shared_ptr<ComposedField>> m_composedFields;

};

#endif // GLOBALFIELDFUNCTION_H
