#ifndef _INITIALIZATION_
#define _INITIALIZATION_

void InitializeMatrix(double **Matrix,int m,int n,double InitialValue);

void InitializeArray(double *Array,int m,double InitialValue);

void InitializeArray_Complex(cufftDoubleComplex *Array,int Dim,double InitialValue_R,double InitialValue_I);

#endif
