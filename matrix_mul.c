#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void generate_random_matrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = rand() % 10;  // Random values between 0 and 9
        }
    }
}

void print_matrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) {
            printf("Usage: mpirun -np <num_processes> ./matrix_mult <rows_A> <cols_A> <cols_B>\n");
        }
        MPI_Finalize();
        return 1;
    }

    int rows_A = atoi(argv[1]);
    int cols_A = atoi(argv[2]);
    int cols_B = atoi(argv[3]);
    
    int *A = NULL;
    int *B = NULL;
    int *C = (int *)malloc(rows_A * cols_B * sizeof(int));
    
    if (rank == 0) {
        A = (int *)malloc(rows_A * cols_A * sizeof(int));
        B = (int *)malloc(cols_A * cols_B * sizeof(int));
        
        generate_random_matrix(A, rows_A, cols_A);
        generate_random_matrix(B, cols_A, cols_B);

        printf("Matrix A (%dx%d):\n", rows_A, cols_A);
        print_matrix(A, rows_A, cols_A);
        printf("Matrix B (%dx%d):\n", cols_A, cols_B);
        print_matrix(B, cols_A, cols_B);
    }

    // Broadcast the dimensions of the matrices
    MPI_Bcast(&cols_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols_B, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Scatter rows of A to all processes
    int rows_per_process = rows_A / size;
    int *local_A = (int *)malloc(rows_per_process * cols_A * sizeof(int));
    MPI_Scatter(A, rows_per_process * cols_A, MPI_INT, local_A, rows_per_process * cols_A, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast matrix B to all processes
    if (rank != 0) {
        B = (int *)malloc(cols_A * cols_B * sizeof(int));
    }
    MPI_Bcast(B, cols_A * cols_B, MPI_INT, 0, MPI_COMM_WORLD);

    // Local computation of C
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < cols_B; j++) {
            C[i * cols_B + j] = 0;
            for (int k = 0; k < cols_A; k++) {
                C[i * cols_B + j] += local_A[i * cols_A + k] * B[k * cols_B + j];
            }
        }
    }

    // Gather the results back to the root process
    MPI_Gather(C, rows_per_process * cols_B, MPI_INT, C, rows_per_process * cols_B, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Result Matrix C (%dx%d):\n", rows_A, cols_B);
        print_matrix(C, rows_A, cols_B);
        free(A);
        free(B);
    }

    free(local_A);
    free(C);
    MPI_Finalize();
    return 0;
}

