#include <gtest/gtest.h>
#include "mesh.h"


//--------------------------------------------------------------------------
// Using same vertices in all tests
//--------------------------------------------------------------------------
std::vector<glm::vec3> verts{   glm::vec3(0.0f, 0.0f, 0.0f),
                                glm::vec3(-1.0f, 0.0f, -1.0f),
                                glm::vec3(1.0f, 0.0f, -1.0f),
                                glm::vec3(-1.0f, 0.0f, 1.0f),
                                glm::vec3(1.0f, 0.0f, 1.0f)     };

std::vector<glm::vec3> norms{   glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f)),
                                glm::normalize(glm::vec3(-1.0f, 1.0f, -1.0f)),
                                glm::normalize(glm::vec3(1.0f, 1.0f, -1.0f)),
                                glm::normalize(glm::vec3(-1.0f, 1.0f, 1.0f)),
                                glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f))     };

//  3-----4
//  |\ 2 /|
//  | \ / |
//  |3 0 1|
//  | / \ |
//  |/ 0 \|
//  1-----2

//--------------------------------------------------------------------------

TEST(MeshTest, OneRingNeighbourOrderVertsCCW)
{
    // Initialise test data


    std::vector<glm::ivec3> tris;
    tris.push_back(glm::ivec3(0, 1, 2));
    tris.push_back(glm::ivec3(0, 2, 4));
    tris.push_back(glm::ivec3(0, 4, 3));
    tris.push_back(glm::ivec3(0, 3, 1));


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
    std::vector<int> expectedOneRingVertex0{1, 2, 4, 3};


    EXPECT_EQ(expectedOneRingVertex0.size(), vertsOneRing[0].size());
    EXPECT_EQ(expectedOneRingVertex0, vertsOneRing[0]);
}

//--------------------------------------------------------------------------

TEST(MeshTest, OneRingNeighbourUnOrderVertsCCW)
{
    // Initialise test data


    std::vector<glm::ivec3> tris;
    tris.push_back(glm::ivec3(0, 3, 4));
    tris.push_back(glm::ivec3(0, 2, 1));
    tris.push_back(glm::ivec3(2, 0, 4));
    tris.push_back(glm::ivec3(1, 3, 0));


    // Set Mesh data
    Mesh mesh;
    mesh.m_meshVerts = verts;
    mesh.m_meshNorms = norms;
    mesh.m_meshTris = tris;


    // Get one ring neighbourhood
    std::vector<std::vector<int>> vertsOneRing;
    mesh.ComputeOneRing();
    mesh.GetOneRingNeighours(vertsOneRing);


    // Expected result of vertex 0 one ring - ccw starting from rightmost(CW) of first triangle
    std::vector<int> expectedOneRingVertex0{4, 3, 1, 2};


    EXPECT_EQ(expectedOneRingVertex0.size(), vertsOneRing[0].size());
    EXPECT_EQ(expectedOneRingVertex0, vertsOneRing[0]);
}

//--------------------------------------------------------------------------

TEST(MeshTest, OneRingNeighbourOrderVertsCW)
{
    // Initialise test data


    std::vector<glm::ivec3> tris;
    tris.push_back(glm::ivec3(0, 1, 2));
    tris.push_back(glm::ivec3(0, 2, 4));
    tris.push_back(glm::ivec3(0, 4, 3));
    tris.push_back(glm::ivec3(0, 3, 1));


    // Set Mesh data
    Mesh mesh;
    mesh.m_meshVerts = verts;
    mesh.m_meshNorms = norms;
    mesh.m_meshTris = tris;


    // Get one ring neighbourhood
    std::vector<std::vector<int>> vertsOneRing;
    mesh.ComputeOneRing(false);
    mesh.GetOneRingNeighours(vertsOneRing);


    // Expected result of vertex 0 one ring - CW reverse of CCW
    std::vector<int> expectedOneRingVertex0{3, 4, 2, 1};


    EXPECT_EQ(expectedOneRingVertex0.size(), vertsOneRing[0].size());
    EXPECT_EQ(expectedOneRingVertex0, vertsOneRing[0]);
}

//--------------------------------------------------------------------------

TEST(MeshTest, OneRingNeighbourUnOrderVertsCW)
{
    // Initialise test data


    std::vector<glm::ivec3> tris;
    tris.push_back(glm::ivec3(0, 3, 4));
    tris.push_back(glm::ivec3(0, 2, 1));
    tris.push_back(glm::ivec3(2, 0, 4));
    tris.push_back(glm::ivec3(1, 3, 0));


    // Set Mesh data
    Mesh mesh;
    mesh.m_meshVerts = verts;
    mesh.m_meshNorms = norms;
    mesh.m_meshTris = tris;


    // Get one ring neighbourhood
    std::vector<std::vector<int>> vertsOneRing;
    mesh.ComputeOneRing(false);
    mesh.GetOneRingNeighours(vertsOneRing);


    // Expected result of vertex 0 one ring - cw reverse of ccw
    std::vector<int> expectedOneRingVertex0{2, 1, 3, 4};


    EXPECT_EQ(expectedOneRingVertex0.size(), vertsOneRing[0].size());
    EXPECT_EQ(expectedOneRingVertex0, vertsOneRing[0]);
}


// MAIN ///////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
