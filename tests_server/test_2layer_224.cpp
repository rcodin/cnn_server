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

using namespace std;

int main() {
	//224x224x3 Conv
		//load network weights
	size_t bytes = sizeof(float);
	int alignment = bytes * 8;
	//create input 
	//conv1->relu->pool

		//Conv11
		Conv_conf conv11_conf = {3, 3, 1, 1};
		Data_conf input11_conf = {224, 224, 3};
		Data_conf output11_conf = {224, 224, 64};

		
		//conv12
		Conv_conf conv12_conf = {3, 3, 1, 1};
		Data_conf input12_conf = {224, 224, 64};
		Data_conf output12_conf = {224, 224, 64};

		//Pool1
		Pool_conf pool1_conf = {2, 2};
		Data_conf input13_conf = {224, 224, 64};
		Data_conf output13_conf = {112, 112, 64};

	float *input11 = (float *)mkl_calloc(input11_conf.h * input11_conf.w * input11_conf.c , bytes, alignment);
	float *output11 = (float *)mkl_calloc(output11_conf.h * output11_conf.w * output11_conf.c , bytes, alignment);
	float *output12 = (float *)mkl_calloc(output12_conf.h * output12_conf.w * output12_conf.c , bytes, alignment);
	float *output13 = (float *)mkl_calloc(output13_conf.h * output13_conf.w * output13_conf.c , bytes, alignment);

	float *conv11_weights;
	float *conv12_weights;

	float *conv11_biases;
	float *conv12_biases;

	std::string weight_dir = "/home/roni/project/files/vgg_16/tensorflow/weights_dir/";
    std::string image_file = "/home/roni/project/files/vgg_16/tensorflow/laska.png";

	Image_cfg input_cfg = {224, 224};
    // float *input = (float *)malloc(input_cfg.rows * input_cfg.cols * 3);

    int err = read_image_rgb(image_file, input_cfg, input11);

	cnpy::NpyArray arr11 = cnpy::npy_load(weight_dir+"conv1_1_W.npy");
	conv11_weights = arr11.data<float>();

	cnpy::NpyArray arr12 = cnpy::npy_load(weight_dir+"conv1_2_W.npy");
	conv12_weights = arr12.data<float>();

	// load conv biases
    cnpy::NpyArray arr11_biases = cnpy::npy_load(weight_dir+"conv1_1_b.npy");
	conv11_biases = arr11_biases.data<float>();

	cnpy::NpyArray arr12_biases = cnpy::npy_load(weight_dir+"conv1_2_b.npy");
	conv12_biases = arr12_biases.data<float>();

	//Group 1

	int h_num_tiles = 8;
	int w_num_tiles = 8;

	for (int h_idx = 0; h_idx < h_num_tiles; h_idx++) {
		for (int w_idx = 0; w_idx < w_num_tiles; w_idx++) {
			TILE_IDX input_tile_idx = {h_idx, w_idx};

			// conv_im2row(input11, output11, conv11_weights,conv11_biases, conv11_conf, input11_conf, output11_conf);
			// conv_im2row(output11, output12, conv12_weights, conv12_biases, conv12_conf, input12_conf, output12_conf);
			// pool_forward(output12, output13, input13_conf, input21_conf,pool1_conf);
		}
	}
}