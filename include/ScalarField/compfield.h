#ifndef COMPFIELD_H
#define COMPFIELD_H

#include <cuda_runtime.h>

/// @author Idris Miles
/// @version 1.0

class CompField
{
public:
    __host__ __device__ CompField(int _fieldFuncA = -1, int _fieldFuncB = -1, int _compOp = -1) :
        fieldFuncA(_fieldFuncA),
        fieldFuncB(_fieldFuncB),
        compOp(_compOp)
    {
    }

    int fieldFuncA;
    int fieldFuncB;
    int compOp;

};

#endif // COMPFIELD_H
