#ifndef MESH_H
#define MESH_H

#include <vector>
#include <glm/glm.hpp>


struct VertexBoneData
{
    unsigned int boneID[4];
    float boneWeight[4];
};


class Mesh
{
public:
    Mesh();


    // Mesh info
    std::vector<glm::vec3> m_meshVerts;
    std::vector<glm::vec3> m_meshNorms;
    std::vector<glm::ivec3> m_meshTris;
    std::vector<VertexBoneData> m_meshBoneWeights;
    glm::vec3 m_colour;

    // Rig info
    std::vector<glm::vec3> m_rigVerts;
    std::vector<glm::vec3> m_rigNorms;
    std::vector<VertexBoneData> m_rigBoneWeights;
    std::vector<glm::vec3> m_rigJointColours;
};

#endif // MESH_H
