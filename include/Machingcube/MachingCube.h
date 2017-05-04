#ifndef MACHINGCUBE_H_
#define MACHINGCUBE_H_
//----------------------------------------------------------------------------------------------------------------------
/// @file MachineCube.h
/// @brief basic maching cube algorithm
//----------------------------------------------------------------------------------------------------------------------

#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <glm/glm.hpp>
#include <cmath>



#include <cuda_runtime.h>
#include <cuda.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtx/vector_angle.hpp>

//----------------------------------------------------------------------------------------------------------------------
/// @class MachingCube "include/MachingCube/MachingCube.h"
/// @brief basic maching cube algorithm
/// @author Xiaosong Yang, Idris Miles
/// @version 1.0
/// @date 14/01/13, 18/04/2017
//----------------------------------------------------------------------------------------------------------------------


// code from http://paulbourke.net/geometry/polygonise/

typedef struct {
   glm::vec3    p[8];
   float        val[8];
} Voxel;

typedef struct {
    glm::vec3    p[3];         /* Vertices */
} Triangle;

// code finished

class MachingCube
{
public :

    //----------------------------------------------------------------------------------------------------------------------
    /// @brief default constructor
    //----------------------------------------------------------------------------------------------------------------------
    MachingCube();

    //----------------------------------------------------------------------------------------------------------------------
    /// @brief default destructor
    //----------------------------------------------------------------------------------------------------------------------
    ~MachingCube();


    //----------------------------------------------------------------------------------------------------------------------
    /// @brief polygonize the iso surface
    //----------------------------------------------------------------------------------------------------------------------
    static void Polygonize(std::vector<glm::vec3> &_verts, std::vector<glm::vec3> &_norms, float *_volumeData, const float &_isolevel, const int &_w, const int &_h, const int &_d, const float &_voxelW = 1.0f, const float &_voxelH = 1.0f, const float &_voxelD = 1.0f);


    //----------------------------------------------------------------------------------------------------------------------
    /// @brief polygonize the iso surface on the GPU,
    /// probably not the most optimal as it's basically a cut and paste job of the CPU implementation,
    /// but still much faster than the CPU version
    //----------------------------------------------------------------------------------------------------------------------
    static void PolygonizeGPU(std::vector<glm::vec3> &_verts, std::vector<glm::vec3> &_norms, float *_volumeData, const float &_isolevel, const int &_w, const int &_h, const int &_d, const float &_voxelW = 1.0f, const float &_voxelH = 1.0f, const float &_voxelD = 1.0f);

protected :

    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Generate vertices and normals
    //----------------------------------------------------------------------------------------------------------------------
    static void createVerts(std::vector<glm::vec3> &_verts, std::vector<glm::vec3> &_norms, float *_volumeData, const float &_isolevel, const int &_w, const int &_h, const int &_d, const float &_voxelW, const float &_voxelH, const float &_voxelD);

    //----------------------------------------------------------------------------------------------------------------------
    /// @brief extract triangles from each voxel, add the triangles into tri vector
    //----------------------------------------------------------------------------------------------------------------------
    static unsigned int MachingTriangles(Voxel g, float iso, std::vector<Triangle> &triList);

    //----------------------------------------------------------------------------------------------------------------------
    /// @brief intepolate the intersection point from the level value
    //----------------------------------------------------------------------------------------------------------------------
    static glm::vec3 VertexInterp(float isolevel, glm::vec3 p1, glm::vec3 p2, float valp1, float valp2);

    //----------------------------------------------------------------------------------------------------------------------
    /// @brief compute the normal from the three vertices
    //----------------------------------------------------------------------------------------------------------------------
    static glm::vec3 computeTriangleNormal(Triangle &itr);

};

#endif
//----------------------------------------------------------------------------------------------------------------------

