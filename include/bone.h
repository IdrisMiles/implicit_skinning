#ifndef BONE_H
#define BONE_H

#include <vector>
#include <string>
#include <memory>

#include <glm/glm.hpp>

#include "include/boneAnim.h"


class Bone
{
public:
    Bone();
    ~Bone();

    std::shared_ptr<Bone> m_parent;
    std::vector<std::shared_ptr<Bone>> m_children;

    std::string m_name;
    uint m_boneID;
    glm::mat4 m_transform;
    glm::mat4 m_boneOffset;
    glm::mat4 m_currentTransform;

    std::shared_ptr<BoneAnim> m_boneAnim;
};

#endif // BONE_H
