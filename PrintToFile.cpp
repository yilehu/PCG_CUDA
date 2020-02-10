#include <stdio.h>

void PrintArray(double *Array,char *Directory,char *ArrayName,int n)
{
	FILE *fp;
	fp = fopen(Directory, "a+");

	fprintf(fp,"Array's name is: %s, Number of rows = %d\n",ArrayName,n);
	printf("Array's name is: %s, Number of rows = %d\n",ArrayName,n);

	int i;
	for(i=0;i<n;i++)
	{
		fprintf(fp,"%16.11lf\n",Array[i]);
		printf("%16.11lf\n",Array[i]);
	}
	fclose(fp);
}

void PrintMatrix(double **Matrix,char *Directory,char *MatrixName,int m,int n)
{
	FILE *fp;
	fp = fopen(Directory, "w+");

	fprintf(fp,"Matrix's name is: %s, Number of rows = %d, Number of Columns = %d\n",MatrixName,m,n);
	printf("Matrix's name is: %s, Number of rows = %d, Number of Columns = %d\n",MatrixName,m,n);
	int i,j;
	for(i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
		{
			fprintf(fp,"%16.11lf",Matrix[i][j]);
			printf("%16.11lf",Matrix[i][j]);
		}
		fprintf(fp,"\n");
		printf("\n");
	}
	fclose(fp);
}
