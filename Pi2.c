#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

double rand_double(unsigned int *seed)
{
    return (double)rand_r(seed) / RAND_MAX;
}

int main()
{
    const size_t N = 1000000;
    double step;
    double pi, x, y;
    unsigned int seed;

    int num_threads = omp_get_max_threads();
    double *sums = (double *)calloc(num_threads, sizeof(double));

    step = 1. / (double)N;
    seed = time(NULL);

#pragma omp parallel shared(sums) private(x, y)
    {
        unsigned int seed_private = seed + omp_get_thread_num();
        int id = omp_get_thread_num();
        double sum_private = 0.;

#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i)
        {
            x = rand_double(&seed_private);
            y = rand_double(&seed_private);

            if (x * x + y * y <= 1.0)
            {
                sum_private += 1.0;
            }
        }

        sums[id] = sum_private;
    }

    double sum = 0.;

    for (int i = 0; i < num_threads; ++i)
    {
        sum += sums[i];
    }

    pi = 4.0 * step * sum;

    printf("pi = %.6f\n", pi);

    free(sums);

    return 0;
}
