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

	Conv_conf conv12_conf = {3, 3, 1, 1};
	Data_conf input12_conf = {224, 224, 64};
	Data_conf output12_conf = {224, 224, 64};

	//Pool1
	Pool_conf pool1_conf = {2, 2};
	Data_conf input13_conf = {224, 224, 64};
	Data_conf output13_conf = {112, 112, 64};

	//Conv21
	Conv_conf conv21_conf = {3, 3, 1, 1};
	Data_conf input21_conf = {112, 112, 64};
	Data_conf output21_conf = {112, 112, 128};

	//Conv22
	Conv_conf conv22_conf = {3, 3, 1, 1};
	Data_conf input22_conf = {112, 112, 128};
	Data_conf output22_conf = {112, 112, 128};

	size_t bytes = sizeof(float);
	int alignment = bytes * 8;

	int h_num_tiles = 14;
	int w_num_tiles = 14;

	Conv_conf conv11_tiled_conf = {3, 3, 1, 0};

	Data_conf input11_tiled_conf = {input11_conf.h/h_num_tiles + (conv11_conf.h - 1),
						input11_conf.w/w_num_tiles  + (conv11_conf.w - 1), input11_conf.c};
	Data_conf output11_tiled_conf = {output11_conf.h/h_num_tiles, output11_conf.w/w_num_tiles, output11_conf.c};


	Conv_conf conv12_tiled_conf = {3, 3, 1, 0};
	Data_conf input12_tiled_conf = {input12_conf.h/h_num_tiles + (conv12_conf.h - 1),
					input12_conf.w/w_num_tiles  + (conv12_conf.w - 1), input12_conf.c};
	Data_conf output12_tiled_conf = {output12_conf.h/h_num_tiles, output12_conf.w/w_num_tiles, output12_conf.c};


	Conv_conf conv21_tiled_conf = {3, 3, 1, 0};
	Data_conf input21_tiled_conf = {input21_conf.h/h_num_tiles + (conv21_conf.h - 1),
					input21_conf.w/w_num_tiles  + (conv21_conf.w - 1), input21_conf.c};
	Data_conf output21_tiled_conf = {output21_conf.h/h_num_tiles, output21_conf.w/w_num_tiles, output21_conf.c};


	Conv_conf conv22_tiled_conf = {3, 3, 1, 0};
	Data_conf input22_tiled_conf = {input22_conf.h/h_num_tiles + (conv22_conf.h - 1),
					input22_conf.w/w_num_tiles  + (conv22_conf.w - 1), input22_conf.c};
	Data_conf output22_tiled_conf = {output22_conf.h/h_num_tiles, output22_conf.w/w_num_tiles, output22_conf.c};

	float *input11 = (float *)mkl_calloc(input11_conf.h * input11_conf.w *
		input11_conf.c, bytes, alignment);

	float *output11 = (float *)mkl_calloc(output11_conf.h * output11_conf.w *
		output11_conf.c, bytes, alignment);

	float *input11_tiled = (float *)mkl_calloc(input11_tiled_conf.h * input11_tiled_conf.w *
		input11_tiled_conf.c, bytes, alignment);

	float *output11_tiled = (float *)mkl_calloc(output11_tiled_conf.h * output11_tiled_conf.w *
		output11_tiled_conf.c, bytes, alignment);



	float *input12 = output11;

	float *output12 = (float *)mkl_calloc(output12_conf.h * output12_conf.w *
		output12_conf.c, bytes, alignment);

	float *input12_tiled = (float *)mkl_calloc(input12_tiled_conf.h * input12_tiled_conf.w *
		input12_tiled_conf.c, bytes, alignment);

	float *output12_tiled = (float *)mkl_calloc(output12_tiled_conf.h * output12_tiled_conf.w *
		output12_tiled_conf.c, bytes, alignment);


	float *input13 = output12;
	float *output13 = (float *)mkl_calloc(output13_conf.h * output13_conf.w *
		output13_conf.c, bytes, alignment);

	float *input21 = output13;

	float *output21 = (float *)mkl_calloc(output21_conf.h * output21_conf.w *
		output21_conf.c, bytes, alignment);

	float *input21_tiled = (float *)mkl_calloc(input21_tiled_conf.h * input21_tiled_conf.w *
		input21_tiled_conf.c, bytes, alignment);

	float *output21_tiled = (float *)mkl_calloc(output21_tiled_conf.h * output21_tiled_conf.w *
		output21_tiled_conf.c, bytes, alignment);


	float *input22 = output21;

	float *output22 = (float *)mkl_calloc(output22_conf.h * output22_conf.w *
		output22_conf.c, bytes, alignment);

	float *input22_tiled = (float *)mkl_calloc(input22_tiled_conf.h * input22_tiled_conf.w *
		input22_tiled_conf.c, bytes, alignment);

	float *output22_tiled = (float *)mkl_calloc(output22_tiled_conf.h * output22_tiled_conf.w *
		output22_tiled_conf.c, bytes, alignment);

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


	float *conv12_weights;
	float *conv12_biases;

	float *conv21_weights;
	float *conv21_biases;

	float *conv22_weights;
	float *conv22_biases;
	
	cnpy::NpyArray arr11 = cnpy::npy_load(weight_dir+"conv1_1_W.npy");
	conv11_weights = arr11.data<float>();

    cnpy::NpyArray arr11_biases = cnpy::npy_load(weight_dir+"conv1_1_b.npy");
	conv11_biases = arr11_biases.data<float>();


	cnpy::NpyArray arr12 = cnpy::npy_load(weight_dir+"conv1_2_W.npy");
	conv12_weights = arr12.data<float>();

    cnpy::NpyArray arr12_biases = cnpy::npy_load(weight_dir+"conv1_2_b.npy");
	conv12_biases = arr12_biases.data<float>();


	cnpy::NpyArray arr21 = cnpy::npy_load(weight_dir+"conv2_1_W.npy");
	conv21_weights = arr21.data<float>();

    cnpy::NpyArray arr21_biases = cnpy::npy_load(weight_dir+"conv2_1_b.npy");
	conv21_biases = arr21_biases.data<float>();

	cnpy::NpyArray arr22 = cnpy::npy_load(weight_dir+"conv2_2_W.npy");
	conv22_weights = arr22.data<float>();

    cnpy::NpyArray arr22_biases = cnpy::npy_load(weight_dir+"conv2_2_b.npy");
	conv22_biases = arr22_biases.data<float>();

	int times = 10;

	bool tiled = false;

	auto start = std::chrono::system_clock::now();
	auto end = std::chrono::system_clock::now();

	if (tiled) {
		// while (1)
		for (int h_tile = 0; h_tile < h_num_tiles; h_tile++) {
			for (int w_tile = 0; w_tile < w_num_tiles; w_tile++) {

				int h_base = h_tile * (input11_tiled_conf.h - (conv11_conf.h - 1));
				int w_base = w_tile * (input11_tiled_conf.w - (conv11_conf.w - 1));

				TILE_BASE tile_base = {h_base, w_base};

				load_tile(input11, input11_conf, tile_base, h_num_tiles, 
							input11_tiled, input11_tiled_conf);

				conv_im2row(input11_tiled, output11_tiled, conv11_weights, conv11_biases, conv11_tiled_conf,
					input11_tiled_conf, output11_tiled_conf);

				save_tile(output11_tiled, output11_tiled_conf, tile_base, output11, output11_conf);

			}
		}
		for (int h_tile = 0; h_tile < h_num_tiles; h_tile++) {
			for (int w_tile = 0; w_tile < w_num_tiles; w_tile++) {

				int h_base = h_tile * (input12_tiled_conf.h - (conv12_conf.h - 1));
				int w_base = w_tile * (input12_tiled_conf.w - (conv12_conf.w - 1));

				TILE_BASE tile_base = {h_base, w_base};

				load_tile(input12, input12_conf, tile_base, h_num_tiles, 
							input12_tiled, input12_tiled_conf);

				conv_im2row(input12_tiled, output12_tiled, conv12_weights, conv12_biases, conv12_tiled_conf,
					input12_tiled_conf, output12_tiled_conf);

				save_tile(output12_tiled, output12_tiled_conf, tile_base, output12, output12_conf);

			}
		}
		
		pool_forward(input13, output13, input13_conf, output13_conf,pool1_conf);

		for (int h_tile = 0; h_tile < h_num_tiles; h_tile++) {
			for (int w_tile = 0; w_tile < w_num_tiles; w_tile++) {

				int h_base = h_tile * (input21_tiled_conf.h - (conv21_conf.h - 1));
				int w_base = w_tile * (input21_tiled_conf.w - (conv21_conf.w - 1));

				TILE_BASE tile_base = {h_base, w_base};

				load_tile(input21, input21_conf, tile_base, h_num_tiles, 
							input21_tiled, input21_tiled_conf);

				conv_im2row(input21_tiled, output21_tiled, conv21_weights, conv21_biases, conv21_tiled_conf,
					input21_tiled_conf, output21_tiled_conf);

				save_tile(output21_tiled, output21_tiled_conf, tile_base, output21, output21_conf);

			}
		}

		double tot_time = 0.0;
		for (int i = 0; i < times; i++) {
			start = std::chrono::system_clock::now();
			for (int h_tile = 0; h_tile < h_num_tiles; h_tile++) {
				for (int w_tile = 0; w_tile < w_num_tiles; w_tile++) {

					int h_base = h_tile * (input22_tiled_conf.h - (conv22_conf.h - 1));
					int w_base = w_tile * (input22_tiled_conf.w - (conv22_conf.w - 1));

					TILE_BASE tile_base = {h_base, w_base};

					load_tile(input22, input22_conf, tile_base, h_num_tiles, 
								input22_tiled, input22_tiled_conf);

					conv_im2row(input22_tiled, output22_tiled, conv22_weights, conv22_biases, conv22_tiled_conf,
						input22_tiled_conf, output22_tiled_conf);

					save_tile(output22_tiled, output22_tiled_conf, tile_base, output22, output22_conf);

				}
			}
			end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_time = end-start;
			tot_time += elapsed_time.count();
		}
		cout<<tot_time/times<<endl;
	}
	else {
		conv_im2row(input11, output11, conv11_weights, conv11_biases, conv11_conf,
				input11_conf, output11_conf);
		
		
		conv_im2row(input12, output12, conv12_weights, conv12_biases, conv12_conf,
				input12_conf, output12_conf);
		
		pool_forward(input13, output13, input13_conf, output13_conf,pool1_conf);
		
		conv_im2row(input21, output21, conv21_weights,conv21_biases, conv21_conf, input21_conf, output21_conf);
		
		double tot_time = 0.0;
		for (int i = 0; i < times; i++) {
			start = std::chrono::system_clock::now();
			conv_im2row(input22, output22, conv22_weights,conv22_biases, conv22_conf, input22_conf, output22_conf);
			end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_time = end-start;
			tot_time += elapsed_time.count();
		}
		cout<<tot_time/times<<endl;
	}


	// std::chrono::duration<double> elapsed_time = end-start;

	// cout<<elapsed_time.count()<<endl;

	// for (int i = 0; i < output22_conf.h ; i++) {
	// 	for (int j = 0; j < output22_conf.w; j++) {
	// 		for (int k = 0; k < output22_conf.c; k++) {
	// 			int idx = (i * output22_conf.w + j) * output22_conf.c + k;
	// 			cout<<fixed<<setprecision(10)<<output22[idx]<<endl;
	// 		}
	// 	}
	// }
}