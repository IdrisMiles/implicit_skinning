#ifndef BONE_H
#define BONE_H

#include <vector>
#include <string>
#include <memory>

#include <glm/glm.hpp>

#include "include/boneAnim.h"

/// @author Idris Miles
/// @version 1.0
/// @date 18/04/2017


/// @class Bone
/// @brief A class representing a single bone in an animated rig.
/// The bone knows its relative position in the rig hierarchy having access to its parent and children.
/// The bone hold all its key frames for a single animation.
class Bone
{
public:
    //---------------------------------------------
    /// @brief default constructor
    Bone(){}

    //---------------------------------------------
    /// @brief default destructor
    ~Bone()
    {
        m_parent = nullptr;

        for(auto &&child : m_children)
        {
            child = nullptr;
        }
        m_children.clear();

        m_boneAnim = nullptr;
    }

    //---------------------------------------------
    /// @brief Pointer to this bones parent bone
    std::shared_ptr<Bone> m_parent;

    /// @brief A vector of pointers to this bones children bones
    std::vector<std::shared_ptr<Bone>> m_children;

    /// @brief This bones name
    std::string m_name;

    /// @brief This bones Id, a useful variable supplied by ASSIMP
    uint m_boneID;

    /// @brief This bones rest transform
    glm::mat4 m_transform;

    /// @brief This bones matrix to transform into bonespace
    glm::mat4 m_boneOffset;

    /// @brief This bones transform matrix, after animation
    glm::mat4 m_currentTransform;

    /// @brief This bones animation, we hold a pointer referencing the animation whiich is stored in the Rig
    std::shared_ptr<BoneAnim> m_boneAnim;
};

#endif // BONE_H
