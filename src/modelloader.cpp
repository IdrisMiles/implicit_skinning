#include "include/modelloader.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <iostream>

ModelLoader::ModelLoader()
{

}


void ModelLoader::LoadModel(std::shared_ptr<Model> _model, const std::string &_file)
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


    glm::mat4 globalInverseTransform = ConvertToGlmMat(scene->mRootNode->mTransformation);
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
        _model->m_rig.m_rootBone->m_parent = NULL;


        unsigned int numChildren = scene->mRootNode->mNumChildren;
        for (unsigned int i=0; i<numChildren; i++)
        {
            CopyRigStructure(_model->m_rig.m_boneNameIdMapping, scene, scene->mRootNode->mChildren[i], _model->m_rig, _model->m_rig.m_rootBone, ConvertToGlmMat(scene->mRootNode->mTransformation));
        }
    }

}




void ModelLoader::InitModelMesh(std::shared_ptr<Model> _model, const aiScene *_scene)
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
                    for(unsigned int w=0; w<4; w++)
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
            _model->m_mesh.m_meshBoneWeights[bw].boneID[0] = 0;
            _model->m_mesh.m_meshBoneWeights[bw].boneID[1] = 0;
            _model->m_mesh.m_meshBoneWeights[bw].boneID[2] = 0;
            _model->m_mesh.m_meshBoneWeights[bw].boneID[3] = 0;

            _model->m_mesh.m_meshBoneWeights[bw].boneWeight[0] = 1.0;
            _model->m_mesh.m_meshBoneWeights[bw].boneWeight[1] = 0.0;
            _model->m_mesh.m_meshBoneWeights[bw].boneWeight[2] = 0.0;
            _model->m_mesh.m_meshBoneWeights[bw].boneWeight[3] = 0.0;
        }
    }
}

void ModelLoader::InitRigMesh(std::shared_ptr<Model> _model, const aiScene *_scene)
{
    glm::mat4 mat = ConvertToGlmMat(_scene->mRootNode->mTransformation) * _model->m_rig.m_globalInverseTransform;

    for (uint i = 0 ; i < _scene->mRootNode->mNumChildren ; i++)
    {
        SetRigVerts(_model, _scene->mRootNode, _scene->mRootNode->mChildren[i], mat, mat);
    }

    if(_model->m_mesh.m_rigVerts.size() % 2)
    {
        for(unsigned int i=0; i<_model->m_mesh.m_rigVerts.size()/2; i++)
        {
            int id = i*2;
            if(_model->m_mesh.m_rigVerts[id] == _model->m_mesh.m_rigVerts[id+1])
            {
                std::cout<<"Repeated joint causing rig issue, removing joint\n";
                _model->m_mesh.m_rigVerts.erase(_model->m_mesh.m_rigVerts.begin()+id);
                _model->m_mesh.m_rigJointColours.erase(_model->m_mesh.m_rigJointColours.begin()+id);
                _model->m_mesh.m_rigBoneWeights.erase(_model->m_mesh.m_rigBoneWeights.begin()+id);

                break;
            }
        }
    }

    std::cout<<"Number of rig verts:\t"<<_model->m_mesh.m_rigVerts.size()<<"\n";

}

void ModelLoader::SetRigVerts(std::shared_ptr<Model> _model, aiNode* _pParentNode, aiNode* _pNode, const glm::mat4 &_parentTransform, const glm::mat4 &_thisTransform)
{
    const std::string parentNodeName(_pParentNode->mName.data);
    const std::string nodeName = _pNode->mName.data;
    bool isBone = _model->m_rig.m_boneNameIdMapping.find(nodeName) != _model->m_rig.m_boneNameIdMapping.end();

    glm::mat4 newThisTransform = ConvertToGlmMat(_pNode->mTransformation) * _thisTransform;
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

void ModelLoader::SetJointVert(std::shared_ptr<Model> _model, const std::string _nodeName, const glm::mat4 &_transform, VertexBoneData &_vb)
{
    if(_model->m_rig.m_boneNameIdMapping.find(_nodeName) != _model->m_rig.m_boneNameIdMapping.end())
    {
        _vb.boneID[0] = _model->m_rig.m_boneNameIdMapping[_nodeName];
        _vb.boneWeight[0] = 1.0f;
        _vb.boneWeight[1] = 0.0f;
        _vb.boneWeight[2] = 0.0f;
        _vb.boneWeight[3] = 0.0f;

        _model->m_mesh.m_rigVerts.push_back(glm::vec3(glm::vec4(0.0f,0.0f,0.0f,1.0f) * _transform));
        _model->m_mesh.m_rigJointColours.push_back(glm::vec3(0.4f, 1.0f, 0.4f));
        _model->m_mesh.m_rigBoneWeights.push_back(_vb);
    }
    else
    {
        std::cout<<"This Node is not a bone, skipping\n";
    }

}


glm::mat4 ModelLoader::ConvertToGlmMat(const aiMatrix4x4 &m)
{

}

void ModelLoader::CopyRigStructure(const std::unordered_map<std::__cxx11::string, unsigned int> &_boneMapping, const aiScene *_aiScene, aiNode *_aiNode, Rig &_rig, std::shared_ptr<Bone> _parentBone, const glm::mat4 &_parentTransform)
{

}

BoneAnim ModelLoader::ConvertToBoneAnim(const aiNodeAnim *_pNodeAnim)
{

}
