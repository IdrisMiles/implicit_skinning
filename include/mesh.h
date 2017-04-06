#ifndef MESH_H
#define MESH_H

#include <vector>
#include <algorithm>
#include <glm/glm.hpp>

/// @author Idris Miles
/// @version 1.0


/// @brief MaxNumBlendWeightsPerVertex, maximum blend weights per vertex.
static const unsigned int MaxNumBlendWeightsPerVertex = 4;


/// @struct VertexBoneData
/// @brief Structure to hold bone IDs and bone weights that influence a vertex.
/// Used for Linear Blend Weight(LBW) skinning
struct VertexBoneData
{
    /// @brief m_boneID, array of bone ID's that influence this vertex.
    unsigned int boneID[MaxNumBlendWeightsPerVertex];

    /// @brief m_boneWeight, array of bone weights that control the influence of bones on this vertex.
    float boneWeight[MaxNumBlendWeightsPerVertex];
};


/// @class Mesh
/// @brief Mesh data structure, holds vertices, triangle indices, normals,
/// vertex colours, mesh colour, vertex UVs, vertex bone weights (for skinning).
class Mesh
{
public :
    //------------------------------------------------------------------------------------

    Mesh()
    {

    }

    //------------------------------------------------------------------------------------

    void GetOneRinigNeighours(std::vector<std::vector<int>> &_oneRing) const
    {
        _oneRing.clear();
        _oneRing.resize(m_meshVerts.size());

        auto addNeighbour = [](std::vector<int> vertNeighs, int vert){
            if(std::find(vertNeighs.begin(), vertNeighs.end(), vert) == vertNeighs.end())
            {
                vertNeighs.push_back(vert);
            }
        };

        for(int f=0; f<m_meshTris.size(); ++f)
        {
            int v0 = m_meshTris[f].x;
            int v1 = m_meshTris[f].y;
            int v2 = m_meshTris[f].z;

            addNeighbour(_oneRing[v0], v1);
            addNeighbour(_oneRing[v0], v2);

            addNeighbour(_oneRing[v1], v0);
            addNeighbour(_oneRing[v1], v2);

            addNeighbour(_oneRing[v2], v0);
            addNeighbour(_oneRing[v2], v1);
        }
    }

    //------------------------------------------------------------------------------------


    /// @brief m_meshVerts, a vector containing all vertices of the mesh.
    std::vector<glm::vec3> m_meshVerts;

    /// @brief m_meshNorms, a vector containing all normals of the mesh.
    std::vector<glm::vec3> m_meshNorms;

    /// @brief m_meshTris, a vector containing all the triangle indices of the mesh.
    std::vector<glm::ivec3> m_meshTris;

    /// @brief m_meshBoneWeights, a vector containing all the vertices bone weight data.
    std::vector<VertexBoneData> m_meshBoneWeights;

    /// @brief m_meshVertColours, a vector containing all the vertex colours of the mesh.
    std::vector<glm::vec3> m_meshVertColours;

    /// @brief m_meshUVs, a vector containing all the vertex UV's of the mesh.
    std::vector<glm::vec2> m_meshUVs;

    /// @brief m_colour, a base colour of the whole mesh, used for simple rendering.
    glm::vec3 m_colour;
};

#endif // MESH_H
