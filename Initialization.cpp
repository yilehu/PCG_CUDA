#include <cufft.h>

void InitializeArray(double *Array,int m,double InitialValue)
{
	int i;
	for(i=0;i<m;i++)
	{
		Array[i] = InitialValue;
	}
}

void InitializeMatrix(double **Matrix,int m,int n,double InitialValue)
{
	int i,j;
	for(i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
		{
			Matrix[i][j] = InitialValue;
		}
	}
}

void InitializeArray_Complex(cufftDoubleComplex *Array,int Dim,double InitialValue_R,double InitialValue_I)
{
	int i;
	for(i=0;i<Dim;i++)
	{
		Array[i].x = InitialValue_R;
		Array[i].y = InitialValue_I;
	}
}
