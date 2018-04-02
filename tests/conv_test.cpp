#include <stdio.h>
#include <cstdlib>
#include <stdint.h>
#include <layers.hpp>
#include <utils.hpp>
#include <mkl.h>
#include <cnpy.hpp>
#include <input.hpp>
#include <iomanip>
#include <fstream>
#include <tiling.hpp>
#include <utility>

using namespace std;

int main() {
	Conv_conf conv11_conf = {3, 3, 1, 1};
	Data_conf input11_conf = {224, 224, 3};
	Data_conf output11_conf = {224, 224, 64};



	size_t bytes = sizeof(float);
	int alignment = bytes * 8;


	float *input11 = (float *)mkl_calloc(input11_conf.h * input11_conf.w *
		input11_conf.c, bytes, alignment);

	float *output11 = (float *)mkl_calloc(output11_conf.h * output11_conf.w *
		output11_conf.c, bytes, alignment);


    std::string weight_dir = "/home/roni/project/files/vgg_16/tensorflow/weights_dir/";
    std::string image_file = "/home/roni/project/files/vgg_16/tensorflow/laska.png";

    Image_cfg input_cfg = {224, 224};
    // float *input = (float *)malloc(input_cfg.rows * input_cfg.cols * 3);

    int err = read_image_rgb(image_file, input_cfg, input11);

    ifstream im_file;

    im_file.open("/home/roni/project/files/vgg_16/tensorflow/image_out.log");

    for (int i = 0; i < (224 * 224 *3); i++) {
    	im_file>>input11[i];
    	// cout<<input11[i];
    }


	float *conv11_weights;
	float *conv11_biases;
	
	cnpy::NpyArray arr11 = cnpy::npy_load(weight_dir+"conv1_1_W.npy");
	conv11_weights = arr11.data<float>();

    cnpy::NpyArray arr11_biases = cnpy::npy_load(weight_dir+"conv1_1_b.npy");
	conv11_biases = arr11_biases.data<float>();


	auto start = std::chrono::system_clock::now();


	conv_forward_bias(input11, output11, conv11_weights, conv11_biases, conv11_conf,
				input11_conf, output11_conf);

	auto end = std::chrono::system_clock::now();

	for (int i = 0; i < output11_conf.h; i++) {
		for (int j = 0; j < output11_conf.h; j++) {
			for (int k = 0; k < output11_conf.c; k++) {
				int idx = (i * output11_conf.w + j) * output11_conf.c + k;
				cout<<fixed<<setprecision(10)<<output11[idx]<<endl;
			}
		}
	}

	std::chrono::duration<double> elapsed_time = end-start;

	// cout<<elapsed_time.count()<<endl;

	// auto start1 = std::chrono::system_clock::now();

	

	// auto end1 = std::chrono::system_clock::now();

	// elapsed_time = end1-start1;

	// cout<<elapsed_time.count()<<endl;
}