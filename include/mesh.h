#ifndef MESH_H
#define MESH_H

#include <vector>
#include <glm/glm.hpp>

static const unsigned int MaxNumBlendWeightsPerVertex = 4;

struct VertexBoneData
{
    unsigned int boneID[MaxNumBlendWeightsPerVertex];
    float boneWeight[MaxNumBlendWeightsPerVertex];
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
