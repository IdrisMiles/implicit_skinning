#ifndef POISSONDISKPOINTSET_H
#define POISSONDISKPOINTSET_H

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

    /// @namespace PoissonDiskPointSet
    /// @brief namespace that holds implementation for poisson disk point set method for sampling
    namespace PoissonDiskPointSet
    {

        /// @brief Method to sample a mesh using Poision Disk Point Set method
        /// @todo Implement
        Mesh SampleMesh(const Mesh &_mesh, const int _numSamples);

    }
}

#endif // POISSONDISKPOINTSET_H
