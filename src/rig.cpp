#include "include/rig.h"
#include <glm/gtx/transform.hpp>

Rig::Rig()
{

}


void Rig::Animate(const float _animationTime)
{
    glm::mat4 identity;

    if(!m_animExists)
    {
        m_boneTransforms.resize(1);
        m_boneTransforms[0] = glm::mat4(1.0);
        return;
    }

    unsigned int numBones = m_boneNameIdMapping.size();
    m_boneTransforms.resize(numBones);

    float timeInTicks = _animationTime * m_ticksPerSecond;
    float animationTime = fmod(timeInTicks, m_animationDuration);

    UpdateBoneHierarchy(animationTime, m_rootBone, identity);
}


void Rig::UpdateBoneHierarchy(const float _animationTime, std::shared_ptr<Bone> _pBone, const glm::mat4& _parentTransform)
{
    if(_pBone == nullptr)
    {
        return;
    }

    int BoneIndex = -1;
    if (m_boneNameIdMapping.find(_pBone->m_name) != m_boneNameIdMapping.end())
    {
        BoneIndex = m_boneNameIdMapping.at(_pBone->m_name);
    }

    // Set defualt to bind pose
    glm::mat4 boneTransform(_pBone->m_transform);

    BoneAnim boneAnim = m_boneAnims[_pBone->m_name];


    // Interpolate scaling and generate scaling transformation matrix
    glm::vec3 scalingVec;
    CalcInterpolatedScaling(scalingVec, _animationTime, boneAnim);
    glm::mat4 scalingMat;
    scalingMat = glm::scale(scalingMat, scalingVec);

    // Interpolate rotation and generate rotation transformation matrix
    glm::quat rotationQ;
    CalcInterpolatedRotation(rotationQ, _animationTime, boneAnim);
    glm::mat4 rotationMat(1.0f);
    rotationMat = glm::mat4_cast(rotationQ);

    // Interpolate translation and generate translation transformation matrix
    glm::vec3 translationVec;
    CalcInterpolatedPosition(translationVec, _animationTime, boneAnim);
    glm::mat4 translationMat;
    translationMat = glm::translate(translationMat, translationVec);

    // Combine the above transformations
    boneTransform =  translationMat * rotationMat * scalingMat;


    glm::mat4 globalTransformation = boneTransform * _parentTransform;


    if (BoneIndex != -1)
    {
        _pBone->m_currentTransform = m_globalInverseTransform * _pBone->m_boneOffset * globalTransformation;
    }

    for (uint i = 0 ; i < _pBone->m_children.size() ; i++)
    {
        UpdateBoneHierarchy(_animationTime, _pBone->m_children[i], globalTransformation);
    }

}




void Rig::CalcInterpolatedRotation(glm::quat& _out, const float _animationTime, const BoneAnim &_boneAnin)
{
    if (_boneAnin.m_rotAnim.size() < 1) {
        glm::mat4 a(1.0f);
        _out = glm::quat_cast(a);
        return;
    }

    // we need at least two values to interpolate...
    if (_boneAnin.m_rotAnim.size() == 1) {
        _out = _boneAnin.m_rotAnim[0].rot;
        return;
    }

    uint RotationIndex = FindRotationKeyFrame(_animationTime, _boneAnin);
    uint NextRotationIndex = (RotationIndex + 1);
    assert(NextRotationIndex < _boneAnin.m_rotAnim.size());
    float DeltaTime = _boneAnin.m_rotAnim[NextRotationIndex].time - _boneAnin.m_rotAnim[RotationIndex].time;
    float Factor = (_animationTime - _boneAnin.m_rotAnim[RotationIndex].time) / DeltaTime;
    //assert(Factor >= 0.0f && Factor <= 1.0f);
    glm::quat StartRotationQ = _boneAnin.m_rotAnim[RotationIndex].rot;
    glm::quat EndRotationQ = _boneAnin.m_rotAnim[NextRotationIndex].rot;
    _out = glm::slerp(StartRotationQ, EndRotationQ, Factor);
    glm::normalize(_out);
}

void Rig::CalcInterpolatedPosition(glm::vec3& _out, const float _animationTime, const BoneAnim &_boneAnim)
{
    if (_boneAnim.m_posAnim.size() < 1) {
        _out = glm::vec3(0.0f, 0.0f, 0.0f);
        return;
    }

    // we need at least two values to interpolate...
    if (_boneAnim.m_posAnim.size() == 1) {
        _out = _boneAnim.m_posAnim[0].pos;
        return;
    }

    uint PositionIndex = FindPositionKeyFrame(_animationTime, _boneAnim);
    uint NextPositionIndex = (PositionIndex + 1);
    assert(NextPositionIndex < _boneAnim.m_posAnim.size());
    float DeltaTime = _boneAnim.m_posAnim[NextPositionIndex].time - _boneAnim.m_posAnim[PositionIndex].time;
    float Factor = (_animationTime - _boneAnim.m_posAnim[PositionIndex].time) / DeltaTime;
    //assert(Factor >= 0.0f && Factor <= 1.0f);
    glm::vec3 startPositionV = _boneAnim.m_posAnim[PositionIndex].pos;
    glm::vec3 endPositionV = _boneAnim.m_posAnim[NextPositionIndex].pos;
    _out = startPositionV + (Factor*(endPositionV-startPositionV));

}

void Rig::CalcInterpolatedScaling(glm::vec3& _out, const float _animationTime, const BoneAnim &_boneAnim)
{
    if (_boneAnim.m_scaleAnim.size() < 1) {
        _out = glm::vec3(1.0f, 1.0f, 1.0f);
        return;
    }

    // we need at least two values to interpolate...
    if (_boneAnim.m_scaleAnim.size() == 1) {
        _out = _boneAnim.m_scaleAnim[0].scale;
        return;
    }

    uint scalingIndex = FindScalingKeyFrame(_animationTime, _boneAnim);
    uint nextScalingIndex = (scalingIndex + 1);
    assert(nextScalingIndex < _boneAnim.m_scaleAnim.size());
    float DeltaTime = _boneAnim.m_scaleAnim[nextScalingIndex].time - _boneAnim.m_scaleAnim[scalingIndex].time;
    float Factor = (_animationTime - (float)_boneAnim.m_scaleAnim[scalingIndex].time) / DeltaTime;
    //assert(Factor >= 0.0f && Factor <= 1.0f);
    glm::vec3 startScalingV = _boneAnim.m_scaleAnim[scalingIndex].scale;
    glm::vec3 endScalingV = _boneAnim.m_scaleAnim[nextScalingIndex].scale;
    _out = startScalingV + (Factor*(endScalingV-startScalingV));

}

uint Rig::FindRotationKeyFrame(const float _animationTime, const BoneAnim &_boneAnim)
{
    assert(_boneAnim.m_rotAnim.size() > 0);

    for (uint i = 0 ; i < _boneAnim.m_rotAnim.size() - 1 ; i++) {
        if (_animationTime < (float)_boneAnim.m_rotAnim[i+1].time)
        {
            return i;
        }
    }

    assert(0);
}

uint Rig::FindPositionKeyFrame(const float _animationTime, const BoneAnim &_boneAnim)
{
    assert(_boneAnim.m_posAnim.size() > 0);

    for (uint i = 0 ; i < _boneAnim.m_posAnim.size() - 1 ; i++) {
        if (_animationTime < (float)_boneAnim.m_posAnim[i + 1].time)
        {
            return i;
        }
    }

    assert(0);
}

uint Rig::FindScalingKeyFrame(const float _animationTime, const BoneAnim &_boneAnim)
{
    assert(_boneAnim.m_scaleAnim.size() > 0);

    for (uint i = 0 ; i < _boneAnim.m_scaleAnim.size() - 1 ; i++) {
        if (_animationTime < (float)_boneAnim.m_scaleAnim[i + 1].time)
        {
            return i;
        }
    }

    assert(0);
}
