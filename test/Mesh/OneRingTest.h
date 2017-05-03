#ifndef _ONERINGTEST__H_
#define _ONERINGTEST__H_

//--------------------------------------------------------------------------

#include "MeshShared.h"

//--------------------------------------------------------------------------

TEST(MeshTest, OneRingNeighbourOrderVertsCCW)
{
    // Set Mesh data
    Mesh mesh;
    mesh.m_meshVerts = verts;
    mesh.m_meshNorms = norms;
    mesh.m_meshTris = tris;


    // Get one ring neighbourhood
    std::vector<std::vector<int>> vertsOneRing;
    mesh.ComputeOneRing();
    mesh.GetOneRingNeighours(vertsOneRing);


    // Expected result of vertex 0 one ring - CCW starting from rightmost(CW) of first triangle
    std::vector<std::vector<int>> expectedOneRingVertex;
    expectedOneRingVertex.push_back(std::vector<int>{1, 2, 4, 3});
    expectedOneRingVertex.push_back(std::vector<int>{2, 0, 3});
    expectedOneRingVertex.push_back(std::vector<int>{4, 0, 1});
    expectedOneRingVertex.push_back(std::vector<int>{1, 0, 4});
    expectedOneRingVertex.push_back(std::vector<int>{3, 0, 2});


    EXPECT_EQ(expectedOneRingVertex.size(), vertsOneRing.size());
    for(int i=0; i<expectedOneRingVertex.size(); ++i)
    {
        EXPECT_EQ(expectedOneRingVertex[i].size(), vertsOneRing[i].size());
        EXPECT_EQ(expectedOneRingVertex[i], vertsOneRing[i]);
    }
}

//--------------------------------------------------------------------------

TEST(MeshTest, OneRingNeighbourUnOrderVertsCCW)
{
    // Initialise test data


    std::vector<glm::ivec3> tris2;
    tris2.push_back(glm::ivec3(0, 3, 4));
    tris2.push_back(glm::ivec3(0, 2, 1));
    tris2.push_back(glm::ivec3(2, 0, 4));
    tris2.push_back(glm::ivec3(1, 3, 0));


    // Set Mesh data
    Mesh mesh;
    mesh.m_meshVerts = verts;
    mesh.m_meshNorms = norms;
    mesh.m_meshTris = tris2;


    // Get one ring neighbourhood
    std::vector<std::vector<int>> vertsOneRing;
    mesh.ComputeOneRing();
    mesh.GetOneRingNeighours(vertsOneRing);


    // Expected result of vertex 0 one ring - ccw starting from rightmost(CW) of first triangle
    std::vector<std::vector<int>> expectedOneRingVertex;
    expectedOneRingVertex.push_back(std::vector<int>{4, 3, 1, 2});
    expectedOneRingVertex.push_back(std::vector<int>{2, 0, 3});
    expectedOneRingVertex.push_back(std::vector<int>{4, 0, 1});
    expectedOneRingVertex.push_back(std::vector<int>{1, 0, 4});
    expectedOneRingVertex.push_back(std::vector<int>{3, 0, 2});


    EXPECT_EQ(expectedOneRingVertex.size(), vertsOneRing.size());
    for(int i=0; i<expectedOneRingVertex.size(); ++i)
    {
        EXPECT_EQ(expectedOneRingVertex[i].size(), vertsOneRing[i].size());
        EXPECT_EQ(expectedOneRingVertex[i], vertsOneRing[i]);
    }
}

//--------------------------------------------------------------------------

TEST(MeshTest, OneRingNeighbourOrderVertsCW)
{
    // Initialise test data


    std::vector<glm::ivec3> tris2;
    tris2.push_back(glm::ivec3(0, 1, 2));
    tris2.push_back(glm::ivec3(0, 2, 4));
    tris2.push_back(glm::ivec3(0, 4, 3));
    tris2.push_back(glm::ivec3(0, 3, 1));


    // Set Mesh data
    Mesh mesh;
    mesh.m_meshVerts = verts;
    mesh.m_meshNorms = norms;
    mesh.m_meshTris = tris2;


    // Get one ring neighbourhood
    std::vector<std::vector<int>> vertsOneRing;
    mesh.ComputeOneRing(false);
    mesh.GetOneRingNeighours(vertsOneRing);


    // Expected result of vertex 0 one ring - CW reverse of CCW
    std::vector<std::vector<int>> expectedOneRingVertex;
    expectedOneRingVertex.push_back(std::vector<int>{3, 4, 2, 1});
    expectedOneRingVertex.push_back(std::vector<int>{3, 0, 2});
    expectedOneRingVertex.push_back(std::vector<int>{1, 0, 4});
    expectedOneRingVertex.push_back(std::vector<int>{4, 0, 1});
    expectedOneRingVertex.push_back(std::vector<int>{2, 0, 3});


    EXPECT_EQ(expectedOneRingVertex.size(), vertsOneRing.size());
    for(int i=0; i<expectedOneRingVertex.size(); ++i)
    {
        EXPECT_EQ(expectedOneRingVertex[i].size(), vertsOneRing[i].size());
        EXPECT_EQ(expectedOneRingVertex[i], vertsOneRing[i]);
    }


}

//--------------------------------------------------------------------------

TEST(MeshTest, OneRingNeighbourUnOrderVertsCW)
{
    // Initialise test data


    std::vector<glm::ivec3> tris2;
    tris2.push_back(glm::ivec3(0, 3, 4));
    tris2.push_back(glm::ivec3(0, 2, 1));
    tris2.push_back(glm::ivec3(2, 0, 4));
    tris2.push_back(glm::ivec3(1, 3, 0));


    // Set Mesh data
    Mesh mesh;
    mesh.m_meshVerts = verts;
    mesh.m_meshNorms = norms;
    mesh.m_meshTris = tris2;


    // Get one ring neighbourhood
    std::vector<std::vector<int>> vertsOneRing;
    mesh.ComputeOneRing(false);
    mesh.GetOneRingNeighours(vertsOneRing);


    // Expected result of vertex 0 one ring - cw reverse of ccw
    std::vector<std::vector<int>> expectedOneRingVertex;
    expectedOneRingVertex.push_back(std::vector<int>{2, 1, 3, 4});
    expectedOneRingVertex.push_back(std::vector<int>{3, 0, 2});
    expectedOneRingVertex.push_back(std::vector<int>{1, 0, 4});
    expectedOneRingVertex.push_back(std::vector<int>{4, 0, 1});
    expectedOneRingVertex.push_back(std::vector<int>{2, 0, 3});


    EXPECT_EQ(expectedOneRingVertex.size(), vertsOneRing.size());
    for(int i=0; i<expectedOneRingVertex.size(); ++i)
    {
        EXPECT_EQ(expectedOneRingVertex[i].size(), vertsOneRing[i].size());
        EXPECT_EQ(expectedOneRingVertex[i], vertsOneRing[i]);
    }
}

//--------------------------------------------------------------------------

#endif //_ONERINGTEST__H_
