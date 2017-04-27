#ifndef POISSONDISKPOINTSET_H
#define POISSONDISKPOINTSET_H

#include "Model/mesh.h"


namespace MeshSampler
{

    namespace PoissonDiskPointSet
    {

        Mesh SampleMesh(const Mesh &_mesh, const int _numSamples);

    }
}

#endif // POISSONDISKPOINTSET_H
