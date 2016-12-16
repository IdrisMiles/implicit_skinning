#include "include/bone.h"

Bone::Bone()
{

}


Bone::~Bone()
{
    m_parent = nullptr;

    for(auto &&child : m_children)
    {
        child = nullptr;
    }
    m_children.clear();

    m_boneAnim = nullptr;
}
