#ifndef MESH_H
#define MESH_H

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <glm/glm.hpp>
#include <iostream>

//#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

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

    Mesh() :
        m_oneRingComputed(false)
    {

    }

    //------------------------------------------------------------------------------------

    bool GetOneRingNeighours(std::vector<std::vector<int>> &_oneRing) const
    {
        if(!m_oneRingComputed)
        {
            std::cout<<"not computed one ring yet\n";
            return false;
        }

        _oneRing = m_meshVertsOneRing;
        return true;
    }

    //------------------------------------------------------------------------------------

    void ComputeOneRing()
    {
        if(m_oneRingComputed){return;}

        // make sure one ring container is empty and correct size
        m_meshVertsOneRing.clear();
        m_meshVertsOneRing.resize(m_meshVerts.size());

        // Because ASSIMP can duplicate verts we need to be able to group
        // Vertex Ids of the same vertex in space
        std::unordered_map<glm::vec3, std::vector<int>> vertexHashIds;
        GenerateVertexHashIds(vertexHashIds);


        // Add 2 vertices from neighbouring faces of each vert (assuming triangle mesh)
        std::vector<std::vector<std::pair<int, int>>> oneRingFaces;
        oneRingFaces.resize(m_meshVerts.size());

        for(int f=0; f<m_meshTris.size(); ++f)
        {
            int v0 = m_meshTris[f].x;
            int v1 = m_meshTris[f].y;
            int v2 = m_meshTris[f].z;

            std::vector<int> commonVertIds = vertexHashIds[m_meshVerts[v0]];
            for(int v=0; v<commonVertIds.size(); ++v)
            {
                AddNeighbourFaces(oneRingFaces[commonVertIds[v]], commonVertIds[v], v1, v2);
            }

            commonVertIds = vertexHashIds[m_meshVerts[v1]];
            for(int v=0; v<commonVertIds.size(); ++v)
            {
                AddNeighbourFaces(oneRingFaces[commonVertIds[v]], commonVertIds[v], v2, v0);
            }

            commonVertIds = vertexHashIds[m_meshVerts[v2]];
            for(int v=0; v<commonVertIds.size(); ++v)
            {
                AddNeighbourFaces(oneRingFaces[commonVertIds[v]], commonVertIds[v], v0, v1);
            }
        }

        // Sort and compress our one ring neighbours to just the verts in order and not faces
        for(int v=0; v<m_meshVerts.size(); ++v)
        {
            SortNeighbours(m_meshVertsOneRing[v],  oneRingFaces[v]);
        }

        m_oneRingComputed = true;
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

    std::vector<std::vector<int>> m_meshVertsOneRing;



private:


    //------------------------------------------------------------------------------------
    bool AddNeighbourVert(std::vector<int> &vertNeighs, int vert) const
    {
        if(std::find(vertNeighs.begin(), vertNeighs.end(), vert) == vertNeighs.end())
        {
            vertNeighs.push_back(vert);
            return true;
        }
        else
        {
            return false;
        }
    }

    //------------------------------------------------------------------------------------

    bool AddNeighbourFaces(std::vector<std::pair<int, int>> &faceNeighs, int vert0, int vert1, int vert2) const
    {
        // Sort out order of one ring
        int v1, v2;
        glm::vec3 edge1 = m_meshVerts[vert1] - m_meshVerts[vert0];
        glm::vec3 edge2 = m_meshVerts[vert2] - m_meshVerts[vert0];
        glm::vec3 normal = m_meshNorms[vert0];
        glm::vec3 cross = glm::cross(edge1, edge2);
        float dot = glm::dot(cross, normal);

        if(dot < 0)
        {
            v1 = vert1;
            v2 = vert2;
        }
        else
        {

            v1 = vert2;
            v2 = vert1;
        }


        // adjacent face
        std::pair<int, int> face1 = std::make_pair(v1, v2);
        std::vector<std::pair<int, int>>::iterator face1Pos = std::find(faceNeighs.begin(), faceNeighs.end(), face1);
        if(face1Pos == faceNeighs.end())
        {
           faceNeighs.push_back(face1);
        }
    }

    //------------------------------------------------------------------------------------

    bool SortNeighbours(std::vector<int> &vertNeighs, std::vector<std::pair<int, int>> &faceNeighs, const bool ccw=true) const
    {
        if(faceNeighs.size() < 1)
        {
            return false;
        }

        bool cyclicOneRing = true;

        auto currFace = faceNeighs.begin();
        vertNeighs.push_back((*currFace).first);
        vertNeighs.push_back((*currFace).second);
        faceNeighs.erase(currFace);

        while(faceNeighs.size() > 0)
        {
            currFace = std::find_if(faceNeighs.begin(), faceNeighs.end(), [currFace](std::pair<int, int> f)->bool{
                    return f.first == currFace->second;
                    // f.second = currFace->second;
            });

            if(currFace == faceNeighs.end())
            {
                // not a cyclic one ring, uh oh!
                cyclicOneRing = false;
                currFace = faceNeighs.begin();
            }

            vertNeighs.push_back(currFace->second);
            faceNeighs.erase(currFace);
        }

        if(!ccw)
        {
            std::reverse(vertNeighs.begin(), vertNeighs.end());
        }

        return cyclicOneRing;

    }

    //------------------------------------------------------------------------------------

    void GenerateVertexHashIds(std::unordered_map<glm::vec3, std::vector<int>> &vertexHashIds)
    {
        for(int i=0; i<m_meshVerts.size(); ++i)
        {
            vertexHashIds[m_meshVerts[i]].push_back(i);
        }
    }

    //------------------------------------------------------------------------------------


    bool m_oneRingComputed;

};

#endif // MESH_H
