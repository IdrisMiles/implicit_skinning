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
    std::vector<glm::vec3> m_meshVertColours;
    std::vector<glm::vec2> m_meshUVs;
    glm::vec3 m_colour;
};

#endif // MESH_H
