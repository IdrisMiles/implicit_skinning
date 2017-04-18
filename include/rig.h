#ifndef RIG_H
#define RIG_H

#include <vector>
#include <unordered_map>
#include <memory>

#include "include/boneAnim.h"
#include "include/bone.h"


/// @author Idris Miles
/// @version 1.0
/// @date 18/04/2017


/// @class Rig
/// @brief The class hold an animated rig.
/// @brief Many of the concepts used here are from: http://ogldev.atspace.co.uk/www/tutorial38/tutorial38.html
class Rig
{
public:

    /// @brief defualt constructor
    Rig();

    /// @brief destructor
    ~Rig();

    /// @brief Animate method, animates all the bones in the rig hierarchy at the specified time
    /// @param _animationTime : current time to evaluate the rigs animation
    void Animate(const float _animationTime);

private:
    //--------------------------------------------------
    // Animation related

    /// @brief Method to update the bone hierarchy transforms
    void UpdateBoneHierarchy(const float _animationTime, std::shared_ptr<Bone> _pBone, const glm::mat4& _parentTransform);

    /// @brief Method to interpolate the rotation animation at the request animation time
    void CalcInterpolatedRotation(glm::quat& _out, const float _animationTime, const BoneAnim &_boneAnin);

    /// @brief Method to interpolate the position animation at the request animation time
    void CalcInterpolatedPosition(glm::vec3& _out, const float _animationTime, const BoneAnim &_boneAnim);

    /// @brief Method to interpolate the scaling animation at the request animation time
    void CalcInterpolatedScaling(glm::vec3& _out, const float _animationTime, const BoneAnim &_boneAnim);

    /// @brief Method to find the keyframe closest to the request animation time in the rotation animation keyframes
    uint FindRotationKeyFrame(const float _animationTime, const BoneAnim &_pBoneAnim);

    /// @brief Method to find the keyframe closest to the request animation time in the position animation keyframes
    uint FindPositionKeyFrame(const float _animationTime, const BoneAnim &_pBoneAnim);

    /// @brief Method to find the keyframe closest to the request animation time in the scaling animation keyframes
    uint FindScalingKeyFrame(const float _animationTime, const BoneAnim &_pBoneAnim);


public:

    /// @brief A boolean to check whether animation exists in this rig
    bool m_animExists;

    /// @brief The number of ticks (frames) per second
    float m_ticksPerSecond;

    /// @brief The duration of the aimation in ticks
    float m_animationDuration;

    /// @brief The root bone of the rig
    std::shared_ptr<Bone> m_rootBone;

    /// @brief A map of bone names and pointers to the corresponding bone
    std::unordered_map<std::string, std::shared_ptr<Bone>> m_bones;

    /// @brief A map of bone names and the corresponding bones animations
    std::unordered_map<std::string, BoneAnim> m_boneAnims;

    /// @brief A vector of bone transforms
    std::vector<glm::mat4> m_boneTransforms;

    /// @brief A map of bone names and their bone Ids from ASSIMP
    std::unordered_map<std::string, unsigned int> m_boneNameIdMapping;

    /// @brief This rigs inverse global transform
    glm::mat4 m_globalInverseTransform;

};

#endif // RIG_H
