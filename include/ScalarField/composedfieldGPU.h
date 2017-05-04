#ifndef COMPFIELD_H
#define COMPFIELD_H

//-------------------------------------------------------------------------------

#include <cuda_runtime.h>


//-------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 18/04/2017
//-------------------------------------------------------------------------------

/// @class ComposedFieldCuda
/// @brief composed field class that can be used in CUDA, holds id of fields to compose and id to operator to compose them with
class ComposedFieldCuda
{
public:
    /// @brief constructor
    __host__ __device__ ComposedFieldCuda(int _fieldFuncA = -1, int _fieldFuncB = -1, int _compOp = -1) :
        fieldFuncA(_fieldFuncA),
        fieldFuncB(_fieldFuncB),
        compOp(_compOp)
    {
    }

    /// @brief field A id
    int fieldFuncA;

    /// @brief field B id
    int fieldFuncB;

    /// @brief composition operator id
    int compOp;

};

//-------------------------------------------------------------------------------

#endif // COMPFIELD_H
