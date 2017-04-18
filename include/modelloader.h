#ifndef MODELLOADER_H
#define MODELLOADER_H


// ASSIMP includes
#include <assimp/scene.h>
#include <assimp/matrix4x4.h>

#include "include/model.h"


//-------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 18/04/2017
//-------------------------------------------------------------------------------


/// @class ModelLoader
/// @brief This class loades a model from file
class ModelLoader
{
public:
    /// @brief constructor
    ModelLoader();

    /// @brief Method to load a new model from file
    /// @param _file : file we wish to load
    static Model *LoadModel(const std::string &_file);

private:

    /// @brief Method to initialise mesh
    static void InitModelMesh(Model *_model, const aiScene *_scene);

    /// @brief Method to initialise rig mesh
    static void InitRigMesh(Model* _model, const aiScene *_scene);

    /// @brief Method to initialise the rig
    static void InitRig(Model* _model, const aiScene *_scene);

    static void SetRigVerts(Model* _model, aiNode *_pParentNode, aiNode *_pNode, const glm::mat4 &_parentTransform, const glm::mat4 &_thisTransform);
    static void SetJointVert(Model* _model, const std::string _nodeName, const glm::mat4 &_transform, VertexBoneData &_vb);

    static glm::mat4 ConvertToGlmMat(const aiMatrix4x4 &m);
    static void CopyRigStructure(const std::unordered_map<std::string, unsigned int> &_boneMapping, const aiScene *_aiScene, aiNode *_aiNode, Rig &_rig, std::shared_ptr<Bone> _parentBone, const glm::mat4 &_parentTransform);
    static BoneAnim ConvertToBoneAnim(const aiNodeAnim *_pNodeAnim);
    static const aiNodeAnim* FindNodeAnim(const aiAnimation* _pAnimation, const std::string _nodeName);

};

#endif // MODELLOADER_H
