#ifndef RIG_H
#define RIG_H

#include <vector>
#include <unordered_map>
#include <memory>

#include "include/boneAnim.h"
#include "include/bone.h"

class Rig
{
public:
    Rig();

    void Animate(const float _animationTime);

private:
    // Animation related
    void UpdateBoneHierarchy(const float _animationTime, std::shared_ptr<Bone> _pBone, const glm::mat4& _parentTransform);
    void CalcInterpolatedRotation(glm::quat& _out, const float _animationTime, const BoneAnim &_boneAnin);
    void CalcInterpolatedPosition(glm::vec3& _out, const float _animationTime, const BoneAnim &_boneAnim);
    void CalcInterpolatedScaling(glm::vec3& _out, const float _animationTime, const BoneAnim &_boneAnim);
    uint FindRotationKeyFrame(const float _animationTime, const BoneAnim &_pBoneAnim);
    uint FindPositionKeyFrame(const float _animationTime, const BoneAnim &_pBoneAnim);
    uint FindScalingKeyFrame(const float _animationTime, const BoneAnim &_pBoneAnim);


public:
    // Attributes
    bool m_animExists;
    unsigned int m_numAnimations;
    unsigned int m_animationID;
    float m_ticksPerSecond;
    float m_animationDuration;

    std::shared_ptr<Bone> m_rootBone;
    std::unordered_map<std::string, std::shared_ptr<Bone>> m_bones;
    std::unordered_map<std::string, BoneAnim> m_boneAnims;
    std::vector<glm::mat4> m_boneTransforms;

    std::unordered_map<std::string, unsigned int> m_boneNameIdMapping;
    glm::mat4 m_globalInverseTransform;

};

#endif // RIG_H
