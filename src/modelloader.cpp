#include "include/modelloader.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <iostream>

ModelLoader::ModelLoader()
{

}


void ModelLoader::LoadModel(Model* _model, const std::string &_file)
{
    const aiScene *scene;
    Assimp::Importer m_importer;

    // Load mesh with ASSIMP
    scene = m_importer.ReadFile(    _file,
                                    aiProcess_GenSmoothNormals |
                                    aiProcess_Triangulate |
                                    aiProcess_JoinIdenticalVertices |
                                    aiProcess_SortByPType   );
    if(!scene)
    {
        std::cout<<"Error loading "<<_file<<" with assimp\n";
        return;
    }


    glm::mat4 globalInverseTransform = glm::mat4(1.0f);//ConvertToGlmMat(scene->mRootNode->mTransformation);
    _model->m_rig.m_globalInverseTransform  = glm::inverse(globalInverseTransform);

    if(!scene)
    {
        std::cout<<"No valid AIScene\n";
    }
    else
    {
        InitModelMesh(_model, scene);
        InitRigMesh(_model, scene);




        _model->m_rig.m_rootBone= std::shared_ptr<Bone>(new Bone());
        _model->m_rig.m_rootBone->m_name = std::string(scene->mRootNode->mName.data);
        _model->m_rig.m_rootBone->m_transform = ConvertToGlmMat(scene->mRootNode->mTransformation);
        _model->m_rig.m_rootBone->m_boneOffset = glm::mat4(1.0f);
        _model->m_rig.m_rootBone->m_parent = nullptr;
        if(_model->m_rig.m_animExists)
        {
            const aiNodeAnim *pNodeAnim = FindNodeAnim(scene->mAnimations[scene->mNumAnimations-1], std::string(scene->mRootNode->mName.data));
            if(pNodeAnim)
            {
                _model->m_rig.m_boneAnims[_model->m_rig.m_rootBone->m_name] = ConvertToBoneAnim(pNodeAnim);
                _model->m_rig.m_rootBone->m_boneAnim = std::make_shared<BoneAnim>(_model->m_rig.m_boneAnims[_model->m_rig.m_rootBone->m_name]);

            }
            else
            {
                BoneAnim rootAnim;
                rootAnim.m_name = _model->m_rig.m_rootBone->m_name;
                rootAnim.m_posAnim.push_back(PosAnim(0.0f, glm::vec3(0, 0, 0)));
                rootAnim.m_scaleAnim.push_back(ScaleAnim(0.0f, glm::vec3(1, 1, 1)));
                _model->m_rig.m_boneAnims[_model->m_rig.m_rootBone->m_name] = rootAnim;
                _model->m_rig.m_rootBone->m_boneAnim = std::make_shared<BoneAnim>(_model->m_rig.m_boneAnims[_model->m_rig.m_rootBone->m_name]);
            }

            unsigned int numChildren = scene->mRootNode->mNumChildren;
            for (unsigned int i=0; i<numChildren; i++)
            {
                CopyRigStructure(_model->m_rig.m_boneNameIdMapping, scene, scene->mRootNode->mChildren[i], _model->m_rig, _model->m_rig.m_rootBone, ConvertToGlmMat(scene->mRootNode->mTransformation));
            }
        }
    }

}




void ModelLoader::InitModelMesh(Model* _model, const aiScene *_scene)
{
    if(_scene->HasMeshes())
    {
        unsigned int nb=0;
        unsigned int indexOffset = 0;
        for(unsigned int i=0; i<_scene->mNumMeshes; i++)
        {
            // Mesh tris/element array
            unsigned int numFaces = _scene->mMeshes[i]->mNumFaces;
            for(unsigned int f=0; f<numFaces; f++)
            {
                auto face = _scene->mMeshes[i]->mFaces[f];
                _model->m_mesh.m_meshTris.push_back(glm::ivec3(face.mIndices[0]+indexOffset, face.mIndices[1]+indexOffset, face.mIndices[2]+indexOffset));
            }

            // Mesh verts and norms
            unsigned int numVerts = _scene->mMeshes[i]->mNumVertices;
            for(unsigned int v=0; v<numVerts; v++)
            {
                auto vert = _scene->mMeshes[i]->mVertices[v];
                auto norm = _scene->mMeshes[i]->mNormals[v];
                _model->m_mesh.m_meshVerts.push_back(glm::vec3(vert.x, vert.y, vert.z));
                _model->m_mesh.m_meshNorms.push_back(glm::vec3(norm.x, norm.y, norm.z));
            }


            _model->m_mesh.m_meshBoneWeights.resize(_model->m_mesh.m_meshVerts.size());

            // Mesh bones
            unsigned int numBones = _scene->mMeshes[i]->mNumBones;
            for(unsigned int b=0; b<numBones; b++)
            {
                auto bone = _scene->mMeshes[i]->mBones[b];
                unsigned int boneIndex = 0;
                std::string boneName = bone->mName.data;

                // Check this is a new bone
                if(_model->m_rig.m_boneNameIdMapping.find(boneName) == _model->m_rig.m_boneNameIdMapping.end())
                {
                    boneIndex = nb;
                    nb++;
                    _model->m_rig.m_boneNameIdMapping[boneName] = boneIndex;
                }
                else
                {
                    boneIndex = _model->m_rig.m_boneNameIdMapping[boneName];
                }


                // Bone vertex weights
                unsigned int boneWeights = bone->mNumWeights;
                for(unsigned int bw=0; bw<boneWeights; bw++)
                {
                    unsigned int vertexID = indexOffset + bone->mWeights[bw].mVertexId;
                    float vertexWeight = bone->mWeights[bw].mWeight;
                    for(unsigned int w=0; w<MaxNumBlendWeightsPerVertex; w++)
                    {
                        if(_model->m_mesh.m_meshBoneWeights[vertexID].boneWeight[w] < FLT_EPSILON)
                        {
                            _model->m_mesh.m_meshBoneWeights[vertexID].boneWeight[w] = vertexWeight;
                            _model->m_mesh.m_meshBoneWeights[vertexID].boneID[w] = boneIndex;
                            break;
                        }
                    }
                }

            } // end for numBones

            indexOffset = _model->m_mesh.m_meshVerts.size();

        } // end for numMeshes

    }// end if has mesh


    if(_scene->HasAnimations())
    {
        _model->m_rig.m_animExists = true;
        _model->m_rig.m_numAnimations = _scene->mNumAnimations;
        _model->m_rig.m_animationID = _model->m_rig.m_numAnimations - 1;
        _model->m_rig.m_ticksPerSecond = _scene->mAnimations[_model->m_rig.m_animationID]->mTicksPerSecond;
        _model->m_rig.m_animationDuration = _scene->mAnimations[_model->m_rig.m_animationID]->mDuration;

    }
    else
    {
        _model->m_rig.m_animExists = false;

        for(unsigned int bw=0; bw<_model->m_mesh.m_meshVerts.size();bw++)
        {
            for(unsigned int bwpv = 0; bwpv < MaxNumBlendWeightsPerVertex; bwpv++)
            {
                _model->m_mesh.m_meshBoneWeights[bw].boneID[bwpv] = 0;
                _model->m_mesh.m_meshBoneWeights[bw].boneWeight[bwpv] = 0.0;
            }
        }
    }
}

void ModelLoader::InitRigMesh(Model *_model, const aiScene *_scene)
{
    glm::mat4 mat = ConvertToGlmMat(_scene->mRootNode->mTransformation) * _model->m_rig.m_globalInverseTransform;

    for (uint i = 0 ; i < _scene->mRootNode->mNumChildren ; i++)
    {
        SetRigVerts(_model, _scene->mRootNode, _scene->mRootNode->mChildren[i], mat, mat);
    }

    if(_model->m_rigMesh.m_meshVerts.size() % 2)
    {
        for(unsigned int i=0; i<_model->m_rigMesh.m_meshVerts.size()/2; i++)
        {
            int id = i*2;
            if(_model->m_rigMesh.m_meshVerts[id] == _model->m_rigMesh.m_meshVerts[id+1])
            {
                std::cout<<"Repeated joint causing rig issue, removing joint\n";
                _model->m_rigMesh.m_meshVerts.erase(_model->m_rigMesh.m_meshVerts.begin()+id);
                _model->m_rigMesh.m_meshVertColours.erase(_model->m_rigMesh.m_meshVertColours.begin()+id);
                _model->m_rigMesh.m_meshBoneWeights.erase(_model->m_rigMesh.m_meshBoneWeights.begin()+id);

                break;
            }
        }
    }

    std::cout<<"Number of rig verts:\t"<<_model->m_rigMesh.m_meshVerts.size()<<"\n";

}

void ModelLoader::SetRigVerts(Model *_model, aiNode* _pParentNode, aiNode* _pNode, const glm::mat4 &_parentTransform, const glm::mat4 &_thisTransform)
{
    const std::string parentNodeName(_pParentNode->mName.data);
    const std::string nodeName = _pNode->mName.data;
    bool isBone = _model->m_rig.m_boneNameIdMapping.find(nodeName) != _model->m_rig.m_boneNameIdMapping.end();

    glm::mat4 newThisTransform = _thisTransform * ConvertToGlmMat(_pNode->mTransformation);
    glm::mat4 newParentTransform = _parentTransform;
    aiNode* newParent = _pParentNode;

    VertexBoneData v2;

    if(isBone)
    {
        // parent joint
        SetJointVert(_model, parentNodeName, _parentTransform, v2);

        // This joint
        SetJointVert(_model, nodeName, newThisTransform, v2);

        // This joint becomes new parent
        newParentTransform = newThisTransform;
        newParent = _pNode;
    }


    // Repeat for rest of the joints
    for (uint i = 0 ; i < _pNode->mNumChildren ; i++)
    {
        SetRigVerts(_model, newParent, _pNode->mChildren[i], newParentTransform, newThisTransform);
    }
}

void ModelLoader::SetJointVert(Model *_model, const std::string _nodeName, const glm::mat4 &_transform, VertexBoneData &_vb)
{
    if(_model->m_rig.m_boneNameIdMapping.find(_nodeName) != _model->m_rig.m_boneNameIdMapping.end())
    {
        _vb.boneID[0] = _model->m_rig.m_boneNameIdMapping[_nodeName];
        _vb.boneWeight[0] = 1.0f;
        _vb.boneWeight[1] = 0.0f;
        _vb.boneWeight[2] = 0.0f;
        _vb.boneWeight[3] = 0.0f;

        _model->m_rigMesh.m_meshVerts.push_back(glm::vec3(_transform*glm::vec4(0.0f,0.0f,0.0f,1.0f)));
        _model->m_rigMesh.m_meshVertColours.push_back(glm::vec3(0.4f, 1.0f, 0.4f));
        _model->m_rigMesh.m_meshBoneWeights.push_back(_vb);
    }
    else
    {
        std::cout<<"This Node is not a bone, skipping\n";
    }

}


glm::mat4 ModelLoader::ConvertToGlmMat(const aiMatrix4x4 &m)
{
//    glm::mat4 a(  m.a1, m.a2, m.a3, m.a4,
//                  m.b1, m.b2, m.b3, m.b4,
//                  m.c1, m.c2, m.c3, m.c4,
//                  m.d1, m.d2, m.d3, m.d4);

    glm::mat4 a(  m.a1, m.b1, m.c1, m.d1,
                  m.a2, m.b2, m.c2, m.d2,
                  m.a3, m.b3, m.c3, m.d3,
                  m.a4, m.b4, m.c4, m.d4);
    return a;
}


const aiBone* GetBone(const aiScene *_aiScene, std::string _name)
{
    for(unsigned int i=0; i<_aiScene->mNumMeshes; i++)
    {
        for(unsigned int j=0; j<_aiScene->mMeshes[i]->mNumBones; j++)
        {
            if(std::string(_aiScene->mMeshes[i]->mBones[j]->mName.data) == _name)
            {
                return _aiScene->mMeshes[i]->mBones[j];
            }
        }
    }

    return NULL;
}

const aiNodeAnim* ModelLoader::FindNodeAnim(const aiAnimation* _pAnimation, const std::string _nodeName)
{
    for (uint i = 0 ; i < _pAnimation->mNumChannels ; i++) {
        const aiNodeAnim* pNodeAnim = _pAnimation->mChannels[i];

        if (std::string(pNodeAnim->mNodeName.data) == _nodeName) {
            return pNodeAnim;
        }
    }

    return NULL;
}

void ModelLoader::CopyRigStructure(const std::unordered_map<std::string, unsigned int> &_boneMapping, const aiScene *_aiScene, aiNode *_aiNode, Rig &_rig, std::shared_ptr<Bone> _parentBone, const glm::mat4 &_parentTransform)
{
    std::shared_ptr<Bone> newBone = std::shared_ptr<Bone>(new Bone());
    glm::mat4 newParentTransform = _parentTransform;

    // Check if this is a bone
    if(_boneMapping.find(std::string(_aiNode->mName.data)) != _boneMapping.end())
    {
        newBone->m_name = std::string(_aiNode->mName.data);
        newBone->m_transform = _parentTransform * ConvertToGlmMat(_aiNode->mTransformation);

        // Get bone offset matrix
        const aiBone* paiBone = GetBone(_aiScene, newBone->m_name);
        if(paiBone)
        {
            std::cout<<std::string(paiBone->mName.data)<< "Model::CopyRigStructure | valid bone\n";
            newBone->m_boneOffset = ConvertToGlmMat(paiBone->mOffsetMatrix);
        }
        else
        {
            std::cout<<"Model::CopyRigStructure | Well sheet, didn't find a bone in aiScene with name matching: "<<newBone->m_name<<". Thus not boneOffsetMatrix\n";
            newBone->m_boneOffset = glm::mat4(1.0f);
        }

        // Get animation data
        const aiNodeAnim *pNodeAnim = FindNodeAnim(_aiScene->mAnimations[_aiScene->mNumAnimations-1], newBone->m_name);
        if(pNodeAnim)
        {
            std::cout<<std::string(pNodeAnim->mNodeName.data)<<"Model::CopyRigStructure | valid nodeAnim\n";
            _rig.m_boneAnims[newBone->m_name] = ConvertToBoneAnim(pNodeAnim);
            newBone->m_boneAnim = std::make_shared<BoneAnim>(_rig.m_boneAnims[newBone->m_name]);

        }
        else
        {
            std::cout<<"Model::CopyRigStructure | Daaannnng, didn't find aiNodeAnim in aiAnimation["<<_aiScene->mNumAnimations<<"] with matching name: "<<newBone->m_name<<". Thus No animation.\n";
            BoneAnim blankAnim;
            blankAnim.m_name = newBone->m_name;
            blankAnim.m_posAnim.push_back(PosAnim(0.0f, glm::vec3(0, 0, 0)));
            blankAnim.m_scaleAnim.push_back(ScaleAnim(0.0f, glm::vec3(1, 1, 1)));
            _rig.m_boneAnims[newBone->m_name] = blankAnim;
            newBone->m_boneAnim = std::make_shared<BoneAnim>(_rig.m_boneAnims[newBone->m_name]);
        }

        // Set parent and set child
        newBone->m_parent = _parentBone;
        _parentBone->m_children.push_back(newBone);

        newParentTransform = glm::mat4(1.0f);
    }
    else
    {
        // forward on this nodes transform to affect the next bone
        std::cout<<"Model::CopyRigStructure | "<<std::string(_aiNode->mName.data)<<" Is not a Bone, probably an arbitrary transform.\n";
        newBone = _parentBone;
        newParentTransform = _parentTransform * ConvertToGlmMat(_aiNode->mTransformation);
    }


    unsigned int numChildren = _aiNode->mNumChildren;
    for (unsigned int i=0; i<numChildren; i++)
    {
        CopyRigStructure(_boneMapping, _aiScene, _aiNode->mChildren[i], _rig, newBone, newParentTransform);
    }
}

BoneAnim ModelLoader::ConvertToBoneAnim(const aiNodeAnim *_pNodeAnim)
{
    BoneAnim newBoneAnim;

    if(_pNodeAnim != NULL)
    {
        // Rotation animation
        for(unsigned int i=0; i<_pNodeAnim->mNumRotationKeys; i++)
        {
            float time = _pNodeAnim->mRotationKeys[i].mTime;
            glm::quat rot;
            rot.x = _pNodeAnim->mRotationKeys[i].mValue.x;
            rot.y = _pNodeAnim->mRotationKeys[i].mValue.y;
            rot.z = _pNodeAnim->mRotationKeys[i].mValue.z;
            rot.w = _pNodeAnim->mRotationKeys[i].mValue.w;
            RotAnim rotAnim = {time, rot};
            newBoneAnim.m_rotAnim.push_back(rotAnim);
        }

        // Position animation
        for(unsigned int i=0; i<_pNodeAnim->mNumPositionKeys; i++)
        {
            float time = _pNodeAnim->mPositionKeys[i].mTime;
            glm::vec3 pos;
            pos.x = _pNodeAnim->mPositionKeys[i].mValue.x;
            pos.y = _pNodeAnim->mPositionKeys[i].mValue.y;
            pos.z = _pNodeAnim->mPositionKeys[i].mValue.z;
            PosAnim posAnim = {time, pos};
            newBoneAnim.m_posAnim.push_back(posAnim);
        }

        // Scaling animation
        for(unsigned int i=0; i<_pNodeAnim->mNumScalingKeys; i++)
        {
            float time = _pNodeAnim->mScalingKeys[i].mTime;
            glm::vec3 scale;
            scale.x = _pNodeAnim->mScalingKeys[i].mValue.x;
            scale.y = _pNodeAnim->mScalingKeys[i].mValue.y;
            scale.z = _pNodeAnim->mScalingKeys[i].mValue.z;
            ScaleAnim scaleAnim = {time, scale};
            newBoneAnim.m_scaleAnim.push_back(scaleAnim);
        }
    }
    else
    {
        std::cout<<"Invalid aiNodeAnim\n";
    }

    return newBoneAnim;
}
