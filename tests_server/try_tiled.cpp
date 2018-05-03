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

	int h_num_tiles = 56;
	int w_num_tiles = 56;

	Conv_conf conv11_tiled_conf = {3, 3, 1, 0};
	Data_conf input11_tiled_conf = {input11_conf.h/h_num_tiles + (conv11_conf.h - 1),
						input11_conf.w/w_num_tiles  + (conv11_conf.w - 1), input11_conf.c};
	Data_conf output11_tiled_conf = {output11_conf.h/h_num_tiles, output11_conf.w/w_num_tiles, output11_conf.c};

	float *input11 = (float *)mkl_calloc(input11_conf.h * input11_conf.w *
		input11_conf.c, bytes, alignment);

	float *output11 = (float *)mkl_calloc(output11_conf.h * output11_conf.w *
		output11_conf.c, bytes, alignment);

	float *input11_tiled;// = (float *)mkl_calloc(input11_tiled_conf.h * input11_tiled_conf.w *
		// input11_tiled_conf.c, bytes, alignment);

	float *output11_tiled;// = (float *)mkl_calloc(output11_tiled_conf.h * output11_tiled_conf.w *
		// output11_tiled_conf.c, bytes, alignment);

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

	bool tiled = false;

	auto start = std::chrono::system_clock::now();

	if (tiled) {
		// while (1)
		for (int h_tile = 0; h_tile < h_num_tiles; h_tile++) {
			for (int w_tile = 0; w_tile < w_num_tiles; w_tile++) {

				int h_base = h_tile * (input11_tiled_conf.h - 2);
				int w_base = w_tile * (input11_tiled_conf.w - 2);

				// cout<<h_base<<" "<<w_base<<endl;
				for (int out_h_idx = 0; out_h_idx < input11_tiled_conf.h; out_h_idx++) {
			    	for (int out_w_idx = 0; out_w_idx < input11_tiled_conf.w; out_w_idx++) {
			    		for (int c_idx = 0; c_idx < input11_tiled_conf.c; c_idx++) {
			    			int in_h_idx = h_base + out_h_idx - 1;
			    			int in_w_idx = w_base + out_w_idx - 1;

			    			int out_idx = (out_h_idx * input11_tiled_conf.w + out_w_idx) * input11_tiled_conf.c + c_idx;
				    		int in_idx = (in_h_idx * input11_conf.w + in_w_idx) * input11_conf.c + c_idx;

				    		// if (h_tile == 0 && w_tile == 1) {
				    		// 	cout<<out_idx<<" "<<in_idx<<endl;
				    		// }
				    		if (in_h_idx >= 0 && in_h_idx < input11_conf.h && in_w_idx >= 0 && in_w_idx < input11_conf.w)
			    				input11_tiled[out_idx] = input11[in_idx];
			    			else {
			    				input11_tiled[out_idx] = 0;
				    			// cout<<in_h_idx<<"	"<<in_w_idx<<endl;
			    			}
			    		}
			    	}
			    }

				conv_im2row(input11_tiled, output11_tiled, conv11_weights, conv11_biases, conv11_tiled_conf,
					input11_tiled_conf, output11_tiled_conf);


				for (int h_idx = 0; h_idx < output11_tiled_conf.h; h_idx++) {
			    	for (int w_idx = 0; w_idx < output11_tiled_conf.w; w_idx++) {
			    		for (int c_idx = 0; c_idx < output11_tiled_conf.c; c_idx++) {

			    			int in_idx = (h_idx * output11_tiled_conf.w + w_idx) * output11_tiled_conf.c + c_idx;
			    			int out_idx = ((h_idx + h_base) * output11_conf.w + (w_idx + w_base)) * output11_conf.c + c_idx;

			    			output11[out_idx] = output11_tiled[in_idx];
			    			output11_tiled[in_idx] = 0;
			    		}
			    	}
			    }
			}
		}

	}
	else {
		conv_im2row(input11, output11, conv11_weights, conv11_biases, conv11_conf,
				input11_conf, output11_conf);
	}

	auto end = std::chrono::system_clock::now();

	// for (int i = 0; i < output11_conf.h; i++) {
	// 	for (int j = 0; j < output11_conf.h; j++) {
	// 		for (int k = 0; k < output11_conf.c; k++) {
	// 			int idx = (i * output11_conf.w + j) * output11_conf.c + k;
	// 			cout<<fixed<<setprecision(10)<<output11[idx]<<endl;
	// 		}
	// 	}
	// }

	std::chrono::duration<double> elapsed_time = end-start;

	cout<<elapsed_time.count()<<endl;

	// auto start1 = std::chrono::system_clock::now();

	

	// auto end1 = std::chrono::system_clock::now();

	// elapsed_time = end1-start1;

	// cout<<elapsed_time.count()<<endl;
}