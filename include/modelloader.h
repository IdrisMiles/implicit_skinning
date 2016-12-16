#ifndef MODELLOADER_H
#define MODELLOADER_H


// ASSIMP includes
#include <assimp/scene.h>
#include <assimp/matrix4x4.h>


#include "include/model.h"

class ModelLoader
{
public:
    ModelLoader();

    static void LoadModel(std::shared_ptr<Model> _model, const std::string &_file);
    static void InitModelMesh(std::shared_ptr<Model> _model, const aiScene *_scene);
    static void InitRigMesh(std::shared_ptr<Model> _model, const aiScene *_scene);
    static void SetRigVerts(std::shared_ptr<Model> _model, aiNode *_pParentNode, aiNode *_pNode, const glm::mat4 &_parentTransform, const glm::mat4 &_thisTransform);
    static void SetJointVert(std::shared_ptr<Model> _model, const std::string _nodeName, const glm::mat4 &_transform, VertexBoneData &_vb);

    static glm::mat4 ConvertToGlmMat(const aiMatrix4x4 &m);
    static void CopyRigStructure(const std::unordered_map<std::string, unsigned int> &_boneMapping, const aiScene *_aiScene, aiNode *_aiNode, Rig &_rig, std::shared_ptr<Bone> _parentBone, const glm::mat4 &_parentTransform);
    static BoneAnim ConvertToBoneAnim(const aiNodeAnim *_pNodeAnim);

};

#endif // MODELLOADER_H
