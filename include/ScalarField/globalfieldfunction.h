#ifndef GLOBALFIELDFUNCTION_H
#define GLOBALFIELDFUNCTION_H

//-------------------------------------------------------------------------------

#include <ScalarField/compositionop.h>
#include <ScalarField/fieldfunction.h>
#include <ScalarField/composedfield.h>
#include <ScalarField/composedfieldGPU.h>

#include <Model/mesh.h>
#include <MeshSampler/meshsampler.h>

#include <memory>
#include <vector>


//-------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 18/04/2017
//-------------------------------------------------------------------------------



/// @class GlobalFieldFunction
/// @brief
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

    /// @brief method to initlise size of _fieldFuncs array
    void Fit(const int _numMeshParts);

    /// @brief method to generate HRBF centres from mesh and bone joints, stores results in _hrbfCentres
    /// @param _meshPart : the mesh we want to generate HRBF centres from
    /// @param _boneEnds : the start and end joint of the bone
    /// @param _numPoints : the number of HRBF centres we want to create
    /// @param _hrbfCentres : a mesh to store the resulting points.
    void GenerateHRBFCentres(const Mesh &_meshPart,
                             const std::pair<glm::vec3, glm::vec3> &_boneEnds,
                             const int _numPoints,
                             Mesh &_hrbfCentres);

    /// @brief method to generate individual field functions
    /// @param _hrbfCentres : the hrbf centre to be used ot geneate the field function
    /// @param _meshParts: the original mesh of the part we are generating a field from
    /// @param _id : id of the field function we want to generate
    /// @todo Should probably call GenerateHRBFCentres and PrecomputeFieldFunc from within this method so everything is handled at once.
    void GenerateFieldFuncs(const Mesh &_hrbfCentres, const Mesh &_meshPart, const int _id);

    /// @brief method to precompute fields into textures
    /// @param _id : id of field to precompute
    /// @param _res : resolution of textures
    /// @param _dim : dimension of sample space to map to texture space
    void PrecomputeFieldFunc(const int _id, const int _res, const float _dim);

    /// @brief method to generate composition operators and composedd fields to build up global field
    void GenerateGlobalFieldFunc();

    /// @brief add a composed field to the global field
    /// @param _composedField : the composed field to add to the global field
    void AddComposedField(std::shared_ptr<ComposedField> _composedField);

    /// @brief method to field function to the global field
    /// @param _fieldFunc : the field to add to the global field.
    void AddFieldFunction(std::shared_ptr<FieldFunction> _fieldFunc);

    /// @brief method to add composition operator to the global field.
    /// @param _compOp : composition operator to add.
    void AddCompositionOp(std::shared_ptr<CompositionOp> _compOp);

    //--------------------------------------------------------------------
    // Setters

    /// @brief method to set bone transforms for each field
    /// @param _transform : inverse bone transform to transform space before sampling field.
    void SetRigidTransforms(const std::vector<glm::mat4> &_transforms);

    //--------------------------------------------------------------------
    // Getters

    /// @brief method to get field functions
    std::vector<std::shared_ptr<FieldFunction>> &GetFieldFuncs();

    /// @brief method to get composition operators
    std::vector<std::shared_ptr<CompositionOp>> &GetCompOps();

    /// @brief method to get CPU composed fields
    std::vector<std::shared_ptr<ComposedField>> &GetCompFields();

    /// @brief method to get GPU composed fields
    std::vector<ComposedFieldCuda> &GetCompFieldsCuda();

    /// @brief method to get all primitive field cuda textures
    std::vector<cudaTextureObject_t> GetFieldFunc3DTextures();

    /// @brief method to check if the global field has been genrated
    bool IsGlobalFieldInit() const;

private:

    /// @brief vector of composed fields
    std::vector<std::shared_ptr<ComposedField>> m_composedFields;

    /// @brief vector of composed field - GPU friendly
    std::vector<ComposedFieldCuda> m_composedFieldsCuda;

    /// @brief vector of field functions
    std::vector<std::shared_ptr<FieldFunction>> m_fieldFuncs;

    /// @brief vector of composition operators
    std::vector<std::shared_ptr<CompositionOp>> m_compOps;

    /// @brief bool to check if the global field has been generated yet.
    bool m_globalFieldInit;

};

//-------------------------------------------------------------------------------

#endif // GLOBALFIELDFUNCTION_H
