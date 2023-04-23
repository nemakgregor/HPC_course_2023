#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Function to allocate a 2D array
double** create_2D_array(int rows, int cols) {
    double **arr = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        arr[i] = (double *)malloc(cols * sizeof(double));
    }
    return arr;
}

// Function to free a 2D array
void free_2D_array(double **arr, int rows) {
    for (int i = 0; i < rows; i++) {
        free(arr[i]);
    }
    free(arr);
}

// Function to perform the Jacobi method
void jacobi_solver(double **A, double *b, double *x, int n, int max_iter, double tol) {
    double *x_new = (double *)malloc(n * sizeof(double));
    double error = tol + 1;
    int iter = 0;

    while (iter < max_iter && error > tol) {
        error = 0;

        #pragma omp parallel for reduction(max:error)
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    sum += A[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i][i];
            error = fmax(error, fabs(x_new[i] - x[i]));
        }

        for (int i = 0; i < n; i++) {
            x[i] = x_new[i];
        }
        iter++;
    }

    free(x_new);
}

int main() {
    int n = 4; // size of the system
    int max_iter = 1000; // maximum number of iterations
    double tol = 1e-6; // tolerance

    // Define A matrix and b vector
    double **A = create_2D_array(n, n);
    double b[] = {1, 2, 3, 4};

    A[0][0] = 4; A[0][1] = -1; A[0][2] = 0; A[0][3] = 0;
    A[1][0] = -1; A[1][1] = 4; A[1][2] = -1; A[1][3] = 0;
    A[2][0] = 0; A[2][1] = -1; A[2][2] = 4; A[2][3] = -1;
    A[3][0] = 0; A[3][1] = 0; A[3][2] = -1; A[3][3] = 4;

    double *x = (double *)calloc(n, sizeof(double));

    jacobi_solver(A, b, x, n, max_iter, tol);

    printf("Solution:\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %.6f\n", i, x[i]);
    }

    free_2D_array(A, n);
    free(x);
    return 0;
}
