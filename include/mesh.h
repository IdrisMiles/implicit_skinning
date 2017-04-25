#ifndef MESH_H
#define MESH_H

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

/// @author Idris Miles
/// @version 1.0
/// @date 18/04/2017


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

    /// @brief constructor
    Mesh() :
        m_oneRingComputed(false),
        m_bboxComputed(false)
    {

    }

    //------------------------------------------------------------------------------------

    /// @brief Method to get the one ring neighbourhoods
    /// @param _oneRing : This is a vector of vectors of ints and holds the one ring neighbourhood for each vertex by their Id
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

    /// @brief Method to compute the one ring neighbourhood
    void ComputeOneRing(const bool ccw=true)
    {
        if(m_oneRingComputed && m_oneRingCCW==ccw){return;}

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
                AddNeighbourFaces(oneRingFaces[commonVertIds[v]], v0, v1, v2);
            }

            commonVertIds = vertexHashIds[m_meshVerts[v1]];
            for(int v=0; v<commonVertIds.size(); ++v)
            {
                AddNeighbourFaces(oneRingFaces[commonVertIds[v]], v1, v2, v0);
            }

            commonVertIds = vertexHashIds[m_meshVerts[v2]];
            for(int v=0; v<commonVertIds.size(); ++v)
            {
                AddNeighbourFaces(oneRingFaces[commonVertIds[v]], v2, v0, v1);
            }
        }

        // Sort and compress our one ring neighbours to just the verts in order and not faces
        for(int v=0; v<m_meshVerts.size(); ++v)
        {
            SortNeighbours(m_meshVertsOneRing[v],  oneRingFaces[v], vertexHashIds, ccw);
        }

        m_oneRingCCW = ccw;
        m_oneRingComputed = true;
    }

    //------------------------------------------------------------------------------------

    /// @brief Method to internally compute the axis aligned bounding box of this mesh
    /// @param _recompute : boolean to recompute this if it has previously been computed but the mesh may have changed since
    void ComputeBBox(const bool &_recompute = false)
    {
        if(m_bboxComputed && !_recompute)
        {
            return;
        }

        glm::vec3 min(1e10f, 1e10f, 1e10f);
        glm::vec3 max(0.0f, 0.0f, 0.0f);

        for(auto &v : m_meshVerts)
        {
            min.x = min.x < v.x ? min.x : v.x;
            min.y = min.y < v.y ? min.y : v.y;
            min.z = min.z < v.z ? min.z : v.z;

            max.x = max.x > v.x ? max.x : v.x;
            max.y = max.y > v.y ? max.y : v.y;
            max.z = max.z > v.z ? max.z : v.z;
        }

        m_minBBox = min;
        m_maxBBox = max;

        m_bboxComputed = true;
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

    /// @brief m_meshVertsOneRing, the one ring neighbourhood for each vertex by id
    std::vector<std::vector<int>> m_meshVertsOneRing;

    /// @brief m_minBBox, the min half of the axis aligned bounding box
    glm::vec3 m_minBBox;

    /// @brief m_maxBBox, the max half of the axis aligned bounding box
    glm::vec3 m_maxBBox;



private:


    //------------------------------------------------------------------------------------
    /// @brief Method to add a vertex id to a neighbourhood
    /// @param vertNeighs : the current neighbourhood
    /// @param vert : the vertex id to add to the neighbourhood
    bool AddNeighbourVert(std::vector<int> &vertNeighs, const int vert) const
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

    /// @brief Method to add vert1 and vert2 of a face to the face neighbourhood of vert0
    /// @param faceNeigh : the pairs of vertices in the neighbour faces (we assume triangle faces)
    /// @param vert0 : the vertex whose neighbourhood we are adding to
    /// @param vert1 : a vertex in the face we are adding
    /// @param vert2 : a vertex in the face we are adding
    bool AddNeighbourFaces(std::vector<std::pair<int, int>> &faceNeighs, const int vert0, const int vert1, const int vert2) const
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

    /// @brief Method to sort the nieghbours so that the one ring neighbourhood follows a cw/ccw direction
    /// @param vertNeighs : the vertex neighourhood we are adding to/sorting
    /// @param faceNeighs : the vertex face neighbourhood
    /// @param vertexHashIds : ASSIMP sometimes duplicates vertices, so we have a map that has hashed vertex positions with the collection of vertex ids that are duplicated
    /// @param ccw : boolean to set whether we want our one ring to be ccw or cw
    bool SortNeighbours(std::vector<int> &vertNeighs, std::vector<std::pair<int, int>> &faceNeighs, std::unordered_map<glm::vec3, std::vector<int>> &vertexHashIds, const bool ccw=true) const
    {
        // make sure there aren't any repeated verts with different ids otherwise we bloat the one ring.
        for(int i=0; i<faceNeighs.size(); ++i)
        {
            auto face1 = faceNeighs[i];
            auto f11 = std::hash<glm::vec3>{}(m_meshVerts[face1.first]);
            auto f12 = std::hash<glm::vec3>{}(m_meshVerts[face1.second]);
            for(int j=i+1; j<faceNeighs.size(); ++j)
            {
                auto face2 = faceNeighs[j];
                auto f21 = std::hash<glm::vec3>{}(m_meshVerts[face2.first]);
                auto f22 = std::hash<glm::vec3>{}(m_meshVerts[face2.second]);

                bool needToSwap = false;
                if(f11 == f21)
                {
                    faceNeighs[j].first = faceNeighs[i].first;
                    needToSwap = true;
                }
                if(f11 == f22)
                {
                    faceNeighs[j].second = faceNeighs[i].first;
                }
                if(f12 == f21)
                {
                    faceNeighs[j].first = faceNeighs[i].second;
                }
                if(f12 == f22)
                {
                    faceNeighs[j].second = faceNeighs[i].second;
                    needToSwap = true;
                }

                if(needToSwap)
                {
                    int tmp = faceNeighs[j].second;
                    faceNeighs[j].second = faceNeighs[j].first;
                    faceNeighs[j].first = tmp;
                }
            }
        }


        if(faceNeighs.size() < 1)
        {
            return false;
        }

        bool cyclicOneRing = true;

        auto currFaceIt = faceNeighs.begin();
        auto currFace = *currFaceIt;
        vertNeighs.push_back(currFace.first);
        vertNeighs.push_back(currFace.second);
        faceNeighs.erase(currFaceIt);

        while(faceNeighs.size() > 0)
        {
            auto tmp = std::find_if(faceNeighs.begin(), faceNeighs.end(), [currFace](std::pair<int, int> f)->bool{
                    return f.first == currFace.second;
            });

            if(tmp == faceNeighs.end())
            {
                // not a cyclic one ring, uh oh!
                cyclicOneRing = false;
                currFaceIt = faceNeighs.begin();
            }
            else
            {
                currFaceIt = tmp;

            }

            currFace = *currFaceIt;
            if(std::find(vertNeighs.begin(), vertNeighs.end(), currFace.second) == vertNeighs.end())
            {
                vertNeighs.push_back(currFace.second);
            }
            faceNeighs.erase(currFaceIt);

        }


        if(!ccw)
        {
            std::reverse(vertNeighs.begin(), vertNeighs.end());
        }

        return cyclicOneRing;

    }

    //------------------------------------------------------------------------------------

    /// @brief Method to generate hash ids for each vertex, and add duplicated verts into a vector
    /// @param vertexHashIds :
    void GenerateVertexHashIds(std::unordered_map<glm::vec3, std::vector<int>> &vertexHashIds)
    {
        for(int i=0; i<m_meshVerts.size(); ++i)
        {
            vertexHashIds[m_meshVerts[i]].push_back(i);
        }
    }

    //------------------------------------------------------------------------------------

    /// @brief Boolean to check whether the one ring has been computed yet as it's an expensive operation
    bool m_oneRingComputed;

    bool m_oneRingCCW;

    /// @brief boolean to check whether the axis aligned bounding box has been computed yet.
    bool m_bboxComputed;

};

#endif // MESH_H
