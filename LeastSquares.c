#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

// Generate random noise
double noise() {
    return 2.0 * ((double)rand() / RAND_MAX) - 1.0;
}

// Generate data samples (xi, yi) with some noise
void generate_data(double *x, double *y, int n, double a, double b) {
    for (int i = 0; i < n; i++) {
        x[i] = i;
        y[i] = a * x[i] + b + noise();
    }
}

// Normalize the input data
void normalize_data(double *x, int n) {
    double min = x[0], max = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
    }
    for (int i = 0; i < n; i++) {
        x[i] = (x[i] - min) / (max - min);
    }
}

// Compute the cost function
double compute_cost(double *x, double *y, int n, double a, double b) {
    double cost = 0;
    #pragma omp parallel for reduction(+:cost)
    for (int i = 0; i < n; i++) {
        double diff = y[i] - (a * x[i] + b);
        cost += diff * diff;
    }
    return cost / (2 * n);
}

// Implement the Gradient Descent algorithm
void gradient_descent(double *x, double *y, int n, double *a, double *b, double alpha, int max_iter) {
    double a_gradient, b_gradient;
    for (int iter = 0; iter < max_iter; iter++) {
        a_gradient = 0;
        b_gradient = 0;

        #pragma omp parallel for reduction(+:a_gradient, b_gradient)
        for (int i = 0; i < n; i++) {
            double diff = y[i] - (*a * x[i] + *b);
            a_gradient += -x[i] * diff;
            b_gradient += -diff;
        }

        *a -= alpha * a_gradient / n;
        *b -= alpha * b_gradient / n;
    }
}

int main() {
    int n = 10000; // number of data points
    double a_true = 3.0, b_true = 2.0; // true values of a and b
    double a = 0, b = 0; // initial values of a and b
    double alpha = 0.00001; // learning rate
    int max_iter = 100; // maximum number of iterations

    double *x = (double *)malloc(n * sizeof(double));
    double *y = (double *)malloc(n * sizeof(double));

    srand(time(NULL));
    generate_data(x, y, n, a_true, b_true);
    normalize_data(x, n); // normalize the input data

    printf("Initial cost: %f\n", compute_cost(x, y, n, a, b));

    gradient_descent(x, y, n, &a, &b, alpha, max_iter);

    printf("Final cost: %f\n", compute_cost(x, y, n, a, b));
    printf("a = %f, b = %f\n", a, b);

    free(x);
    free(y);

    return 0;
}
