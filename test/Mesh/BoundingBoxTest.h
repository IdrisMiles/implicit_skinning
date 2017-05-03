#ifndef _BOUNDINGBOXTEST__H_
#define _BOUNDINGBOXTEST__H_

//--------------------------------------------------------------------------

#include "MeshShared.h"

//--------------------------------------------------------------------------

TEST(MeshTest, BoundingBoxA)
{
    // initialise test mesh
    Mesh mesh;
    mesh.m_meshVerts = verts;
    mesh.m_meshNorms = norms;
    mesh.m_meshTris = tris;


    // compute bbox
    mesh.ComputeBBox();

    glm::vec3 minBBox = mesh.m_minBBox;
    glm::vec3 maxBBox = mesh.m_maxBBox;


    // check results
    glm::vec3 expectedMinBBox = glm::vec3(-1.0f, 0.0f, -1.0f);
    glm::vec3 expectedMaxBBox = glm::vec3(1.0f, 0.0f, 1.0f);

    EXPECT_EQ(expectedMinBBox, minBBox);
    EXPECT_EQ(expectedMaxBBox, maxBBox);

}


TEST(MeshTest, BoundingBoxB)
{
    // initialise test mesh
    std::vector<glm::vec3> verts2 { glm::vec3(4.2f, -0.3f, 1.4f),
                                    glm::vec3(-3.2f, -0.1f, 10.4f),
                                    glm::vec3(0.2f, 4.3f, -11.2f),
                                    glm::vec3(6.2f, 6.3f, -6.4f),
                                    glm::vec3(2.5f, 6.1f, -2.0f)
                                  };

    Mesh mesh;
    mesh.m_meshVerts = verts2;
    mesh.m_meshNorms = norms;
    mesh.m_meshTris = tris;


    // compute bbox
    mesh.ComputeBBox();

    glm::vec3 minBBox = mesh.m_minBBox;
    glm::vec3 maxBBox = mesh.m_maxBBox;


    // check results
    glm::vec3 expectedMinBBox = glm::vec3(-3.2f, -0.3f, -11.2f);
    glm::vec3 expectedMaxBBox = glm::vec3(6.2f, 6.3f, 10.4f);

    EXPECT_EQ(expectedMinBBox, minBBox);
    EXPECT_EQ(expectedMaxBBox, maxBBox);

}

//--------------------------------------------------------------------------

#endif //_BOUNDINGBOXTEST__H_
