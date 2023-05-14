#include <cuda.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLOCK_SIZE 16
#define FILTER_SIZE 100
#define FILTER_RADIUS FILTER_SIZE/2

__device__ unsigned char find_median(unsigned char* window) 
{
    for(int i = 0; i < FILTER_SIZE*FILTER_SIZE; i++) 
    {
        int min_idx = i;
        for(int j = i+1; j < FILTER_SIZE*FILTER_SIZE; j++) 
            if(window[j] < window[min_idx])
                min_idx = j;

        unsigned char temp = window[i];
        window[i] = window[min_idx];
        window[min_idx] = temp;
    }

    return window[FILTER_SIZE*FILTER_SIZE/2];
}

__global__ void median_filter_kernel(unsigned char* input, unsigned char* output, int width, int height) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < width && idy < height) 
    {
        unsigned char filterValues[FILTER_SIZE * FILTER_SIZE];

        for (int i = 0; i < FILTER_SIZE; ++i) 
        {
            for (int j = 0; j < FILTER_SIZE; ++j) 
            {
                int currentX = min(max(idx + i - FILTER_RADIUS, 0), width - 1);
                int currentY = min(max(idy + j - FILTER_RADIUS, 0), height - 1);

                filterValues[i * FILTER_SIZE + j] = input[currentY * width + currentX];
            }
        }

        output[idy * width + idx] = find_median(filterValues);
    }
}

int main() 
{
    int width, height, channels;

    unsigned char* img = stbi_load("photo.jpg", &width, &height, &channels, 0);

    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, width * height * channels * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * channels * sizeof(unsigned char));

    cudaMemcpy(d_input, img, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    for(int c = 0; c < channels; c++)
        median_filter_kernel<<<grid, block>>>(d_input + c * width * height, d_output + c * width * height, width, height);

    cudaMemcpy(img, d_output, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_jpg("photo_cartoon.jpg", width, height, channels, img, 100);

    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(img);

    return 0;
}
