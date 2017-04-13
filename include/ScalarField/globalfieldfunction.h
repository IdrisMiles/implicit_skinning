#ifndef GLOBALFIELDFUNCTION_H
#define GLOBALFIELDFUNCTION_H

#include <ScalarField/compositionop.h>
#include <ScalarField/fieldfunction.h>
#include <ScalarField/composedfield.h>
#include <ScalarField/compfield.h>

#include <mesh.h>
#include <MeshSampler/meshsampler.h>

#include <memory>
#include <vector>


class GlobalFieldFunction
{
public:
    /// @brief Default constructor
    GlobalFieldFunction();

    /// @brief Destructor
    ~GlobalFieldFunction();

    //--------------------------------------------------------------------
    // Evaluation functions

    /// @brief Public method to evaluate global field function at a sample point.
    /// @param glm::vec3 _x : Sample point to evaulate in global field.
    /// @ret] float : Value of field at sample point.
    float Eval(const glm::vec3 &_x);

    /// @brief Public method to evaluate gradient of global field function at a sample point.
    /// @param glm::vec3 _x : Sample point to evaulate in global field.
    /// @ret] glm::vec3 : Gradient of field at sample point.
    glm::vec3 Grad(const glm::vec3 &_x);

    //--------------------------------------------------------------------
    // Field generation functions

    /// @brief
    void Fit(const int _numMeshParts);

    /// @brief
    void GenerateHRBFCentres(const Mesh &_meshPart,
                             const glm::vec3 &_startJoint,
                             const glm::vec3 &_endJoint,
                             const int _numPoints,
                             Mesh &_hrbfCentres);

    /// @brief
    void GenerateFieldFuncs(const Mesh &_hrbfCentres, const Mesh &_meshPart, const int _id);

    void PrecomputeFieldFunc(const int _id, const int _res, const float _dim);

    /// @brief
    void GenerateGlobalFieldFunc();

    /// @brief
    void AddComposedField(std::shared_ptr<ComposedField> _composedField);

    /// @brief
    void AddFieldFunction(std::shared_ptr<FieldFunction> _fieldFunc);

    /// @brief
    void AddCompositionOp(std::shared_ptr<CompositionOp> _compOp);

    //--------------------------------------------------------------------
    // Setters

    /// @brief
    void SetRigidTransforms(const std::vector<glm::mat4> &_transforms);

    //--------------------------------------------------------------------
    // Getters

    /// @brief
    std::vector<std::shared_ptr<FieldFunction>> &GetFieldFuncs();

    /// @brief
    std::vector<std::shared_ptr<CompositionOp>> &GetCompOps();

    /// @brief
    std::vector<std::shared_ptr<ComposedField>> &GetCompFields();

    /// @brief
    std::vector<ComposedFieldCuda> &GetCompFieldsCuda();

    /// @brief
    std::vector<cudaTextureObject_t> GetFieldFunc3DTextures();

private:

    /// @brief
    std::vector<std::shared_ptr<ComposedField>> m_composedFields;

    /// @brief
    std::vector<ComposedFieldCuda> m_composedFieldsCuda;

    /// @brief
    std::vector<std::shared_ptr<FieldFunction>> m_fieldFuncs;

    /// @brief
    std::vector<std::shared_ptr<CompositionOp>> m_compOps;

};

#endif // GLOBALFIELDFUNCTION_H
