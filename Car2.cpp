#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h> // 1. Include the OpenMP header file

#define RGB_COMPONENT_COLOR 255

struct PPMPixel {
    int red;
    int green;
    int blue;
};

typedef struct{
    int x, y, all;
    PPMPixel * data;
} PPMImage;

void readPPM(const char *filename, PPMImage& img){
    std::ifstream file (filename);
    if (file){
        std::string s;
        int rgb_comp_color;
        file >> s;
        if (s!="P3") {std::cout<< "error in format"<<std::endl; exit(9);}
        file >> img.x >>img.y;
        file >>rgb_comp_color;
        img.all = img.x*img.y;
        std::cout << s << std::endl;
        std::cout << "x=" << img.x << " y=" << img.y << " all=" <<img.all << std::endl;
        img.data = new PPMPixel[img.all];
        for (int i=0; i<img.all; i++){
            file >> img.data[i].red >>img.data[i].green >> img.data[i].blue;
        }

    }else{
        std::cout << "the file:" << filename << "was not found" << std::endl;
    }
    file.close();
}

void writePPM(const char *filename, PPMImage & img){
    std::ofstream file (filename, std::ofstream::out);
    file << "P3"<<std::endl;
    file << img.x << " " << img.y << " "<< std::endl;
    file << RGB_COMPONENT_COLOR << std::endl;

    for(int i=0; i<img.all; i++){
        file << img.data[i].red << " " << img.data[i].green << " " << img.data[i].blue << (((i+1)%img.x ==0)? "\n" : " ");
    }
    file.close();
}

// 2. Implement the shiftColumns function
void shiftColumns(PPMImage &img) {
    #pragma omp parallel for // 3. Use OpenMP to parallelize the shifting process
    for (int i = 0; i < img.y; i++) {
        PPMPixel temp = img.data[i * img.x + img.x - 1];
        for (int j = img.x - 1; j > 0; j--) {
            img.data[i * img.x + j] = img.data[i * img.x + j - 1];
        }
        img.data[i * img.x] = temp;
    }
}

int main(int argc, char *argv[]){
    PPMImage image;
    readPPM("car.ppm", image);

    // 4. Add a loop to save the image with different frequencies to analyze the performance time
    int num_shifts = 100; // Number of shifts to perform
    int save_freq[] = {1, 10, 20, 50}; // Saving frequencies to analyze
    int num_freqs = sizeof(save_freq) / sizeof(save_freq[0]);

    for (int f = 0; f < num_freqs; f++) {
        double start_time = omp_get_wtime();

        for (int i = 0; i < num_shifts; i++) {
            shiftColumns(image);
            if (i % save_freq[f] == 0) {
                std::string filename = "new_car_" + std::to_string(f) + "_" + std::to_string(i) + ".ppm";
                writePPM(filename.c_str(), image);
            }
        }

        double end_time = omp_get_wtime();
        std::cout << "Saving frequency: " << save_freq[f] << ", elapsed time: " << end_time - start_time << " seconds" << std::endl;
    }

    return 0;
}
