#ifndef GLOBALFIELDFUNCTION_H
#define GLOBALFIELDFUNCTION_H

#include <ScalarField/compositionop.h>
#include <ScalarField/fieldfunction.h>
#include <ScalarField/composedfield.h>

#include <mesh.h>
#include <MeshSampler/meshsampler.h>

#include <memory>
#include <vector>


class GlobalFieldFunction
{
public:
    GlobalFieldFunction();
    ~GlobalFieldFunction();


    float Eval(const glm::vec3 &_x);

    glm::vec3 Grad(const glm::vec3 &_x);


    void GenerateHRBFCentres(const Mesh &_meshPart,
                             const glm::vec3 &_startJoint,
                             const glm::vec3 &_endJoint,
                             const int _numPoints,
                             Mesh &_hrbfCentres);

    void GenerateFieldFuncs(const Mesh &_hrbfCentres);

    void GenerateGlobalFieldFunc();


    void AddComposedField(std::shared_ptr<ComposedField> _composedField);

    void AddFieldFunction(std::shared_ptr<FieldFunction> _fieldFunc);

    void AddCompositionOp(std::shared_ptr<CompositionOp> _compOp);

private:

    std::vector<std::shared_ptr<ComposedField>> m_composedFields;

    std::vector<std::shared_ptr<FieldFunction>> m_fieldFuncs;

    std::vector<std::shared_ptr<CompositionOp>> m_compOps;

};

#endif // GLOBALFIELDFUNCTION_H
