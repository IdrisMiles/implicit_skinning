#ifndef COMPFIELD_H
#define COMPFIELD_H


class CompField
{
public:
    CompField(int _fieldFuncA = -1, int _fieldFuncB = -1, int _compOp = -1) :
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
