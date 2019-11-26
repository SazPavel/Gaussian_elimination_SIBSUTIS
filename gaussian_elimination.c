#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <time.h>

int get_chunk(int total, int commsize, int rank)
{
    int n = total;
    int q = n / commsize;
    if(n % commsize)
        q++;
    int r = commsize * q - n;
    int chunk = q;
    if(rank >= commsize - r)
        chunk = q - 1;
    return chunk;
}

int main(int argc, char *argv[])
{
    int n = 300;
    int rank, commsize;
    double t;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0)
    {
        t = MPI_Wtime();
    }
    
    int nrows = get_chunk(n, commsize, rank);
    int *rows = (int *) malloc(sizeof(*rows) * nrows); 
    double *a = (double *) malloc(sizeof(*a) * nrows * n); 
    double *tmp = (double *) malloc(sizeof(*tmp) * n);
    
    for(int i = 0; i < nrows; i++)
    { 
        rows[i] = rank + commsize * i;
        srand(rows[i] * n);
        for (int j = 0; j < n; j++)
            a[i * n + j] = rand() % 100 + 1;
    }
    
    int row = 0;
    
    for(int i = 0; i < n - 1; i++)
    {
        if(i == rows[row])
		{
            MPI_Bcast(&a[row * n], n, MPI_DOUBLE, rank, MPI_COMM_WORLD);
            for (int j = 0; j < n; j++)
                tmp[j] = a[row * n + j];
            row++;
        }else{
            MPI_Bcast(tmp, n, MPI_DOUBLE, i % commsize, MPI_COMM_WORLD);
        }

        for(int j = row; j < nrows; j++)
		{
            double scaling = a[j * n + i] / tmp[i];
            for(int k = i; k < n; k++)
                a[j * n + k] -= scaling * tmp[k];
        }
    }
    
    double sum = gsum = 1;
    for(int i = 0; i < nrows; i++)
        sum *= a[i * n + rows[i]];
    MPI_Reduce(&sum, &gsum, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
    
    if(rank == 0)
	{
        t = MPI_Wtime() - t;
        printf("Gaussian Elimination (MPI): n %d ", n);
		printf("procs %d, time (sec) %.6f tglob = %f\n",  commsize, t, gsum);
    }
    free(tmp);
    free(rows);
    free(a);
    MPI_Finalize();
    
    return 0;
}
