#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cufft.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Initialization.h"
#include "Initialization_GPU.cu"
#include "PrintToFile.h"
#include "MatrixOperation.cu"

//#define BATCH 1

void SelectGPU()
{
	int i;
	cudaDeviceProp prop;
	cudaGetDeviceCount(&i);
	if(i==1)
		printf("    There is %d GPU device on your PC.\n",i);
	else
		printf("    There are %d GPU devices on your PC.\n",i);
	cudaGetDeviceProperties(&prop,0);
	printf("    Device %d is: %s.  Compute capability: %d.%d, SMs = %d\n",0,prop.name,prop.major,prop.minor,prop.multiProcessorCount);
	printf("    maxThreadsPerBlock = %d, maxThreadsDim = [%d,%d,%d], maxGridSize = [%d,%d,%d]\n",prop.maxThreadsPerBlock,prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2],prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
	cudaSetDevice(0);
	printf("    Device %d is chosen.\n\n",0);
}

double SumDouble(double *Array,int n)
{
	double Sum=0.0;
	for(int i=0;i<n;i++)
	{
		Sum += Array[i];
	}
	return Sum;
}

int  main()
{
	int GridDim = 80,BlockDim = 256;
	SelectGPU();
	printf("This is my first CUDA code.\n");

	//************ �ļ���д �������� ***********//
	char *Directory1,*Directory2;
	Directory1 = "Array.txt";
	Directory2 = "Matrix.txt";

	//************ ��ʱ�� �������� ***********//
	int START_CLOCK,END_CLOCK;
	double Iter_Running_Time,Total_Running_Time;

	//************ CG �������� ***********//
	int IterationNum;
	int n = 1000000;
	int Bandwidth = 5;
	double error_old,error,error0 = 1.0e-6;
	double alpha,beta,denominator;

	double *x,*PartialSum;
	x = (double*)malloc(n*sizeof(double));
	PartialSum = (double*)malloc(GridDim*sizeof(double));
	InitializeArray(x,n,0.0);
	InitializeArray(PartialSum,GridDim,0.0);

	double *dev_x,*dev_b,*dev_r,*dev_z,*dev_p,*dev_Ax,*dev_PartialSum;
	cufftDoubleComplex *dev_r_Complex,*dev_z_Complex;
	cudaMalloc((void**)&dev_x,n*sizeof(double));
	cudaMalloc((void**)&dev_b,n*sizeof(double));
	cudaMalloc((void**)&dev_r,n*sizeof(double));
	cudaMalloc((void**)&dev_z,n*sizeof(double));
	cudaMalloc((void**)&dev_p,n*sizeof(double));
	cudaMalloc((void**)&dev_Ax,n*sizeof(double));
	cudaMalloc((void**)&dev_PartialSum,GridDim*sizeof(double));
	cudaMalloc((void**)&dev_r_Complex,n*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&dev_z_Complex,n*sizeof(cufftDoubleComplex));
	InitializeArray_GPU<<<GridDim,BlockDim>>>(dev_x,n,0.0);
	InitializeArray_GPU<<<GridDim,BlockDim>>>(dev_b,n,1.0);
	InitializeArray_GPU<<<GridDim,BlockDim>>>(dev_r,n,0.0);
	InitializeArray_GPU<<<GridDim,BlockDim>>>(dev_z,n,0.0);
	InitializeArray_GPU<<<GridDim,BlockDim>>>(dev_p,n,0.0);
	InitializeArray_GPU<<<GridDim,BlockDim>>>(dev_Ax,n,0.0);
	InitializeArray_GPU<<<GridDim,1>>>(dev_PartialSum,GridDim,0.0);
	InitializeArray_Complex_GPU<<<GridDim,BlockDim>>>(dev_r_Complex,n,0.0,0.0);
	InitializeArray_Complex_GPU<<<GridDim,BlockDim>>>(dev_z_Complex,n,0.0,0.0);

	//************ Define Circulant Preconditioner ***********//
	cufftHandle plan;
	cufftDoubleComplex *c,*dev_c,*dev_Eigenvalue;

	c = (cufftDoubleComplex*)malloc(n*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&dev_c,n*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&dev_Eigenvalue,n*sizeof(cufftDoubleComplex));

	InitializeArray_Complex(c,n,0.0,0.0);
	InitializeArray_Complex_GPU<<<GridDim,BlockDim>>>(dev_c,n,0.0,0.0);
	InitializeArray_Complex_GPU<<<GridDim,BlockDim>>>(dev_Eigenvalue,n,0.0,0.0);

	cufftPlan1d(&plan,n,CUFFT_Z2Z,1);

	for(int i=0;i<Bandwidth;i++)
	{
		for(int j=0;j<n-i;j++)
		{
			c[i].x += Bandwidth - i;
		}
		c[i].x /= (double)n;
		if(i!=0)
		{
			c[n-i].x = c[i].x;
		}
	}


	cudaMemcpy(dev_c,c,n*sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice);
	cufftExecZ2Z(plan,dev_c,dev_Eigenvalue,CUFFT_FORWARD);
	Inverse_Complex<<<GridDim,BlockDim>>>(dev_Eigenvalue,n);

	//cudaMemcpy(c,dev_Eigenvalue,n*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost);
	//for(int i=0;i<n;i++)
	//{
	//	printf("Lamda[%d].x = %lf, Lamda[%d].y = %lf\n",i,c[i].x,i,c[i].y);
	//}

	Total_Running_Time = 0.0;
	//Preconditioned Conjugate Gradient//
	//Initialization//
	InitializeArray_GPU<<<GridDim,BlockDim>>>(dev_x,n,0.0);
	MatrixMultiply_GPU<<<GridDim,BlockDim>>>(dev_x,dev_Ax,n,2*Bandwidth-1,Bandwidth);
	cudaMemcpy(dev_r,dev_b,n*sizeof(double),cudaMemcpyDeviceToDevice);
	/************ Solve Linear System with preconditioner ***********/
	R2Z<<<GridDim,BlockDim>>>(dev_r,dev_r_Complex,n);
	cufftExecZ2Z(plan,dev_r_Complex,dev_c,CUFFT_FORWARD);
	Multiply_Complex<<<GridDim,BlockDim>>>(dev_Eigenvalue,dev_c,n);
	cufftExecZ2Z(plan,dev_c,dev_z_Complex,CUFFT_INVERSE);
	Z2R<<<GridDim,BlockDim>>>(dev_z_Complex,dev_z,n);
	/****************************************************************/
	cudaMemcpy(dev_p,dev_z,n*sizeof(double),cudaMemcpyDeviceToDevice);
	IterationNum = 0;
	Dotproduct_Shared_Reduction<<<GridDim,BlockDim>>>(dev_r,dev_z,dev_PartialSum,n);
	cudaMemcpy(PartialSum,dev_PartialSum,GridDim*sizeof(double),cudaMemcpyDeviceToHost);
	error_old = SumDouble(PartialSum,GridDim);
	error = sqrt(error_old);
	//Iteration//
	while(error>error0)
	{
		START_CLOCK = clock();
		MatrixMultiply_GPU<<<GridDim,BlockDim>>>(dev_p,dev_Ax,n,2*Bandwidth-1,Bandwidth);
		Dotproduct_Shared_Reduction<<<GridDim,BlockDim>>>(dev_p,dev_Ax,dev_PartialSum,n);
		cudaMemcpy(PartialSum,dev_PartialSum,GridDim*sizeof(double),cudaMemcpyDeviceToHost);
		denominator = SumDouble(PartialSum,GridDim);
		alpha = error_old/denominator;
		UpdateSolution<<<GridDim,BlockDim>>>(dev_x,dev_p,dev_r,dev_Ax,alpha,n);
		/************ Solve Linear System with preconditioner ***********/
		R2Z<<<GridDim,BlockDim>>>(dev_r,dev_r_Complex,n);
		cufftExecZ2Z(plan,dev_r_Complex,dev_c,CUFFT_FORWARD);
		Multiply_Complex<<<GridDim,BlockDim>>>(dev_Eigenvalue,dev_c,n);
		cufftExecZ2Z(plan,dev_c,dev_z_Complex,CUFFT_INVERSE);
		Z2R<<<GridDim,BlockDim>>>(dev_z_Complex,dev_z,n);
		/****************************************************************/
		Dotproduct_Shared_Reduction<<<GridDim,BlockDim>>>(dev_r,dev_z,dev_PartialSum,n);
		cudaMemcpy(PartialSum,dev_PartialSum,GridDim*sizeof(double),cudaMemcpyDeviceToHost);
		error = SumDouble(PartialSum,GridDim);
		beta = error/error_old;
		UpdateSearchDirection<<<GridDim,BlockDim>>>(dev_p,dev_z,beta,n);
		error_old = error;
		Dotproduct_Shared_Reduction<<<GridDim,BlockDim>>>(dev_r,dev_r,dev_PartialSum,n);
		cudaMemcpy(PartialSum,dev_PartialSum,GridDim*sizeof(double),cudaMemcpyDeviceToHost);
		error = SumDouble(PartialSum,GridDim);
		error = sqrt(error);
		END_CLOCK = clock();
		Iter_Running_Time = (double)(END_CLOCK - START_CLOCK)/CLOCKS_PER_SEC;
		Total_Running_Time += Iter_Running_Time;
		if(IterationNum%1000 == 0) printf("Iteration number = %d, error = %12E, Iteration time = %12.6lf, Total time = %12.6lf\n",IterationNum,error,Iter_Running_Time,Total_Running_Time);
		IterationNum++;
	}
	printf("Iteration number = %d, error = %12E, Total time = %12.6lf\n",IterationNum,error,Total_Running_Time);

	MatrixMultiply_GPU<<<GridDim,BlockDim>>>(dev_x,dev_Ax,n,2*Bandwidth-1,Bandwidth);
	Residual<<<GridDim,BlockDim>>>(dev_b,dev_Ax,dev_r,n);
	Dotproduct_Shared_Reduction<<<GridDim,BlockDim>>>(dev_r,dev_r,dev_PartialSum,n);
	cudaMemcpy(PartialSum,dev_PartialSum,GridDim*sizeof(double),cudaMemcpyDeviceToHost);
	error = SumDouble(PartialSum,GridDim);
	error = sqrt(error);
	printf("Error = %12E\n",error);

	//PrintArray(a,Directory1,"a",n);
	//PrintArray(b,Directory1,"b",n);

	return 0;
}
