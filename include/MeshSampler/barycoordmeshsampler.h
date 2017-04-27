#ifndef _BARYCOORDMESHSAMPLER__H_
#define _BARYCOORDMESHSAMPLER__H_

#include "Model/mesh.h"


namespace MeshSampler
{

    namespace BaryCoord
    {

        Mesh SampleMesh(const Mesh &_mesh, const int _numSamples);

    }

}

#endif //_BARYCOORDMESHSAMPLER__H_
