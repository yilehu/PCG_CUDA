__global__ void InitializeArray_GPU(double *Array,int Dim,double InitialValue)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while(tid<Dim)
	{
		Array[tid] = InitialValue;
		tid += gridDim.x*blockDim.x;
	}
}

__global__ void InitializeArray_Complex_GPU(cufftDoubleComplex *Array,int Dim,double InitialValue_R,double InitialValue_I)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while(tid<Dim)
	{
		Array[tid].x = InitialValue_R;
		Array[tid].y = InitialValue_I;
		tid += gridDim.x*blockDim.x;
	}
}
