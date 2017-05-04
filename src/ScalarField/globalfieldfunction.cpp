#include "ScalarField/globalfieldfunction.h"
#include <algorithm>

GlobalFieldFunction::GlobalFieldFunction():
    m_globalFieldInit(false)
{

}

//----------------------------------------------------------------------------------------------------

GlobalFieldFunction::~GlobalFieldFunction()
{
    m_composedFields.clear();
}

//----------------------------------------------------------------------------------------------------

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

//----------------------------------------------------------------------------------------------------

glm::vec3 GlobalFieldFunction::Grad(const glm::vec3 &_x)
{
    float h= 0.01f;
    float f = Eval(_x);

    float dx = (Eval(_x + glm::vec3(h, 0.0f, 0.0f)) - f) / h;
    float dy = (Eval(_x + glm::vec3(0.0f, h, 0.0f)) - f) / h;
    float dz = (Eval(_x + glm::vec3(0.0f, 0.0f, h)) - f) / h;

    return glm::vec3(dx, dy, dz);
}

//----------------------------------------------------------------------------------------------------

void GlobalFieldFunction::Fit(const int _numMeshParts)
{
    m_fieldFuncs.resize(_numMeshParts);
}

//----------------------------------------------------------------------------------------------------

void GlobalFieldFunction::GenerateHRBFCentres(const Mesh &_meshPart,
                                              const std::pair<glm::vec3, glm::vec3> &_boneEnds,
                                              const int _numPoints,
                                              Mesh &_hrbfCentres)
{
    // Generate HRBF centre by sampling mesh
    _hrbfCentres = MeshSampler::BaryCoord::SampleMesh(_meshPart, _numPoints);


    // Determine distance of closest point to bone
    glm::vec3 edge = _boneEnds.second - _boneEnds.first;
    float minDist = FLT_MAX;
    for(auto &&tri : _meshPart.m_meshTris)
    {
        glm::vec3 v0 = _meshPart.m_meshVerts[tri.x];
        glm::vec3 v1 = _meshPart.m_meshVerts[tri.y];
        glm::vec3 v2 = _meshPart.m_meshVerts[tri.z];

        glm::vec3 e = v0 - _boneEnds.first;
        float t = glm::dot(e, edge);
        float dist = glm::distance(v0, _boneEnds.first + (t*edge));
        minDist = dist < minDist ? dist : minDist;

        e = v1 - _boneEnds.first;
        t = glm::dot(e, edge);
        dist = glm::distance(v1, _boneEnds.first + (t*edge));
        minDist = dist < minDist ? dist : minDist;

        e = v2 - _boneEnds.first;
        t = glm::dot(e, edge);
        dist = glm::distance(v2, _boneEnds.first + (t*edge));
        minDist = dist < minDist ? dist : minDist;
    }


    // Add these points to close holes of scalar field smoothly
    _hrbfCentres.m_meshVerts.push_back(_boneEnds.first - (minDist * glm::normalize(edge)));
    _hrbfCentres.m_meshNorms.push_back(-glm::normalize(edge));
    _hrbfCentres.m_meshVerts.push_back(_boneEnds.second + (minDist * glm::normalize(edge)));
    _hrbfCentres.m_meshNorms.push_back(glm::normalize(edge));


    for(int i=0; i<_hrbfCentres.m_meshVerts.size(); i++)
    {
        glm::vec3 v = _hrbfCentres.m_meshVerts[i];
        float f = glm::dot(v-_boneEnds.first, _boneEnds.second-_boneEnds.first) / glm::dot(_boneEnds.second-_boneEnds.first, _boneEnds.second-_boneEnds.first);
        float h = 0.05f;

        if(f < h && f > 1.0f-h)
        {
            _hrbfCentres.m_meshVerts.erase(_hrbfCentres.m_meshVerts.begin()+i);
            _hrbfCentres.m_meshNorms.erase(_hrbfCentres.m_meshNorms.begin()+i);
            i--;
        }
    }
}

//----------------------------------------------------------------------------------------------------

void GlobalFieldFunction::GenerateFieldFuncs(const Mesh &_hrbfCentres, const Mesh &_meshPart, const int _id)
{
    // Generate HRBF fit and thus scalar field/implicit function
    auto fieldFunc = std::shared_ptr<FieldFunction>(new FieldFunction());
    fieldFunc->Fit(_hrbfCentres.m_meshVerts, _hrbfCentres.m_meshNorms);


    // Find maximun range of scalar field
    float maxDist = FLT_MIN;
    for(auto &&tri : _meshPart.m_meshTris)
    {
        glm::vec3 v0 = _meshPart.m_meshVerts[tri.x];
        glm::vec3 v1 = _meshPart.m_meshVerts[tri.y];
        glm::vec3 v2 = _meshPart.m_meshVerts[tri.z];

        float f0 = fieldFunc->EvalDist(v0);
        maxDist = f0 > maxDist ? f0 : maxDist;
        float f1 = fieldFunc->EvalDist(v1);
        maxDist = f1 > maxDist ? f1 : maxDist;
        float f2 = fieldFunc->EvalDist(v2);
        maxDist = f2 > maxDist ? f2 : maxDist;
    }


    // Set R in order to make field function compactly supported
    fieldFunc->SetSupportRadius(maxDist);

    if(_id < m_fieldFuncs.size())
    {
        m_fieldFuncs[_id] = fieldFunc;
    }
    else
    {
        AddFieldFunction(fieldFunc);
    }
}

//----------------------------------------------------------------------------------------------------

void GlobalFieldFunction::PrecomputeFieldFunc(const int _id, const int _res, const float _dim)
{
    m_fieldFuncs[_id]->PrecomputeField(_res, _dim);
}

//----------------------------------------------------------------------------------------------------

void GlobalFieldFunction::GenerateGlobalFieldFunc()
{
    // Time to build composition tree
    typedef std::shared_ptr<CompositionOp> CompositionOpPtr;

    // Initialise our various type of gradient based operators
    CompositionOpPtr contactOp = CompositionOpPtr(new CompositionOp());
    CompositionOpPtr bulgeOp = CompositionOpPtr(new CompositionOp());


    //TODO: Fit the operators so the dc(alpha) matches specific effect
//    contactOp->SetCompositionOp([](float f1, float f2, float d){
//        if(f1 > 0.7f || f2 > 0.7f) return f1 > f2 ? f1 : f2;

//        auto K = []()

//    });

    contactOp->SetTheta([](float _angleRadians){
        return _angleRadians <= M_PI ? (0.5f*(cosf(_angleRadians)+1.0f)) : 0.0f;
    });

//    bulgeOp->SetCompositionOp([](float f1, float f2, float d){

//    });

    bulgeOp->SetTheta([](float _angleRadians){
        return _angleRadians <= M_PI ? (0.5f*(cosf(2.0f*_angleRadians)+1.0f)) : 1.0f;
    });

    contactOp->Precompute(64);
    bulgeOp->Precompute(64);
    m_compOps.push_back(contactOp);
    m_compOps.push_back(bulgeOp);


    // add composed fields to global field
    for(unsigned int mp=0; mp<m_fieldFuncs.size(); mp+=2)
    {
        int fieldId = 0;
        auto composedField = std::shared_ptr<ComposedField>(new ComposedField());
        composedField->SetCompositionOp(contactOp);
        composedField->SetFieldFunc(m_fieldFuncs[mp], fieldId++);


        if(m_fieldFuncs.size() > mp+1)
        {
            composedField->SetFieldFunc(m_fieldFuncs[mp+1], fieldId++);
        }

        AddComposedField(composedField);
        m_composedFieldsCuda.push_back(ComposedFieldCuda(mp, (fieldId<2)?-1:mp+1, 0));
    }

    m_globalFieldInit = true;
}

//----------------------------------------------------------------------------------------------------

void GlobalFieldFunction::AddComposedField(std::shared_ptr<ComposedField> _composedField)
{
    m_composedFields.push_back(_composedField);
}

//----------------------------------------------------------------------------------------------------

void GlobalFieldFunction::AddFieldFunction(std::shared_ptr<FieldFunction> _fieldFunc)
{
    m_fieldFuncs.push_back(_fieldFunc);
}

//----------------------------------------------------------------------------------------------------

void GlobalFieldFunction::AddCompositionOp(std::shared_ptr<CompositionOp> _compOp)
{
    m_compOps.push_back(_compOp);
}

//----------------------------------------------------------------------------------------------------

void GlobalFieldFunction::SetRigidTransforms(const std::vector<glm::mat4> &_transforms)
{
    for(unsigned int mp=0; mp<_transforms.size(); mp++)
    {
        m_fieldFuncs[mp]->SetTransform(glm::inverse(_transforms[mp]));
    }
}

//----------------------------------------------------------------------------------------------------

std::vector<std::shared_ptr<FieldFunction> > &GlobalFieldFunction::GetFieldFuncs()
{
    return m_fieldFuncs;
}

//----------------------------------------------------------------------------------------------------

std::vector<std::shared_ptr<CompositionOp>> &GlobalFieldFunction::GetCompOps()
{
    return m_compOps;
}

//----------------------------------------------------------------------------------------------------

std::vector<std::shared_ptr<ComposedField>> &GlobalFieldFunction::GetCompFields()
{
    return m_composedFields;
}

//----------------------------------------------------------------------------------------------------

std::vector<ComposedFieldCuda> &GlobalFieldFunction::GetCompFieldsCuda()
{
    return m_composedFieldsCuda;
}

//----------------------------------------------------------------------------------------------------

std::vector<cudaTextureObject_t> GlobalFieldFunction::GetFieldFunc3DTextures()
{
    std::vector<cudaTextureObject_t> textures;
    for(auto f : m_fieldFuncs)
    {
        textures.push_back(f->GetFieldFuncCudaTextureObject());
    }

    return textures;
}

//----------------------------------------------------------------------------------------------------

bool GlobalFieldFunction::IsGlobalFieldInit() const
{
    return m_globalFieldInit;
}
