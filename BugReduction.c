#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

// #define N 100

float dotprod(float * a, float * b, size_t N)
{
    int i, tid;
    float* sum;

    tid = omp_get_thread_num();
    sum = (float *)malloc(sizeof(float));

#pragma omp parallel for// reduction (+: sum)
    for (i = 0; i < N; ++i)
    {
        *sum += a[i] * b[i];
        printf("tid = %d i = %d\n", tid, i);
    }

    return *sum;
}

int main (int argc, char *argv[])
{
    const size_t N = 100;
    int i;
    float* sum;
    
    // float a[N], b[N];
    float * a = (float *)calloc(N, sizeof(float));
    float * b = (float *)calloc(N, sizeof(float));

    for (i = 0; i < N; ++i)
    {
        a[i] = b[i] = (double)i;
    }

    

#pragma omp parallel shared(sum)
    sum = (float *)malloc(sizeof(float));
    *sum = 0.0;
    *sum = dotprod(a, b, N);

    printf("Sum = %f\n", *sum);

    return 0;
}
