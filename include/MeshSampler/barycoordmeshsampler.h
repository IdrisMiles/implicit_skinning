#ifndef _BARYCOORDMESHSAMPLER__H_
#define _BARYCOORDMESHSAMPLER__H_

#include "Model/mesh.h"


//-------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 18/04/2017
//-------------------------------------------------------------------------------


/// @namespace MeshSampeler
/// @brief extending the MeshSampeler namespace
namespace MeshSampler
{


    /// @namespace BaryCoord
    /// @brief namespace that holds implementation for barycentric coordinates method for sampling
    namespace BaryCoord
    {

        /// @brief Method to sample mesh
        /// @param _mesh : Mesh we wish to sample
        Mesh SampleMesh(const Mesh &_mesh, const int _numSamples);

    }

}

#endif //_BARYCOORDMESHSAMPLER__H_
