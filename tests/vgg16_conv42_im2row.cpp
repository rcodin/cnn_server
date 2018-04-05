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

		//Pool2	
		Pool_conf pool2_conf = {2, 2};
		Data_conf input23_conf = {112, 112, 128};
		Data_conf output23_conf = {56, 56, 128};

		float *input23 = output22;
		float *output23 = (float *)mkl_calloc(output23_conf.h * output23_conf.w *
			output23_conf.c, bytes, alignment);

		//Conv31
		Conv_conf conv31_conf = {3, 3, 1, 1};
		Data_conf input31_conf = {56, 56, 128};
		Data_conf output31_conf = {56, 56, 256};

		Conv_conf conv31_tiled_conf = {3, 3, 1, 0};
		Data_conf input31_tiled_conf = {input31_conf.h/h_num_tiles + (conv31_conf.h - 1),
						input31_conf.w/w_num_tiles  + (conv31_conf.w - 1), input31_conf.c};
		Data_conf output31_tiled_conf = {output31_conf.h/h_num_tiles, output31_conf.w/w_num_tiles, output31_conf.c};


		float *input31 = output23;

		float *output31 = (float *)mkl_calloc(output31_conf.h * output31_conf.w *
			output31_conf.c, bytes, alignment);

		float *input31_tiled = (float *)mkl_calloc(input31_tiled_conf.h * input31_tiled_conf.w *
			input31_tiled_conf.c, bytes, alignment);

		float *output31_tiled = (float *)mkl_calloc(output31_tiled_conf.h * output31_tiled_conf.w *
			output31_tiled_conf.c, bytes, alignment);


		//Conv32
		Conv_conf conv32_conf = {3, 3, 1, 1};
		Data_conf input32_conf = {56, 56, 256};
		Data_conf output32_conf = {56, 56, 256};

		Conv_conf conv32_tiled_conf = {3, 3, 1, 0};
		Data_conf input32_tiled_conf = {input32_conf.h/h_num_tiles + (conv32_conf.h - 1),
						input32_conf.w/w_num_tiles  + (conv32_conf.w - 1), input32_conf.c};
		Data_conf output32_tiled_conf = {output32_conf.h/h_num_tiles, output32_conf.w/w_num_tiles, output32_conf.c};


		float *input32 = output31;

		float *output32 = (float *)mkl_calloc(output32_conf.h * output32_conf.w *
			output32_conf.c, bytes, alignment);

		float *input32_tiled = (float *)mkl_calloc(input32_tiled_conf.h * input32_tiled_conf.w *
			input32_tiled_conf.c, bytes, alignment);

		float *output32_tiled = (float *)mkl_calloc(output32_tiled_conf.h * output32_tiled_conf.w *
			output32_tiled_conf.c, bytes, alignment);


		//Conv33
		Conv_conf conv33_conf = {3, 3, 1, 1};
		Data_conf input33_conf = {56, 56, 256};
		Data_conf output33_conf = {56, 56, 256};

		Conv_conf conv33_tiled_conf = {3, 3, 1, 0};
		Data_conf input33_tiled_conf = {input33_conf.h/h_num_tiles + (conv33_conf.h - 1),
						input33_conf.w/w_num_tiles  + (conv33_conf.w - 1), input33_conf.c};
		Data_conf output33_tiled_conf = {output33_conf.h/h_num_tiles, output33_conf.w/w_num_tiles, output33_conf.c};


		float *input33 = output32;

		float *output33 = (float *)mkl_calloc(output33_conf.h * output33_conf.w *
			output33_conf.c, bytes, alignment);

		float *input33_tiled = (float *)mkl_calloc(input33_tiled_conf.h * input33_tiled_conf.w *
			input33_tiled_conf.c, bytes, alignment);

		float *output33_tiled = (float *)mkl_calloc(output33_tiled_conf.h * output33_tiled_conf.w *
			output33_tiled_conf.c, bytes, alignment);

		//Pool3
		Pool_conf pool3_conf = {2, 2};
		Data_conf input34_conf = {56, 56, 256};
		Data_conf output34_conf = {28, 28, 256};

		float *input34 = output33;
		float *output34 = (float *)mkl_calloc(output34_conf.h * output34_conf.w *
			output34_conf.c, bytes, alignment);


		//Conv41
		Conv_conf conv41_conf = {3, 3, 1, 1};
		Data_conf input41_conf = {28, 28, 256};
		Data_conf output41_conf = {28, 28, 512};


		Conv_conf conv41_tiled_conf = {3, 3, 1, 0};
		Data_conf input41_tiled_conf = {input41_conf.h/h_num_tiles + (conv41_conf.h - 1),
						input41_conf.w/w_num_tiles  + (conv41_conf.w - 1), input41_conf.c};
		Data_conf output41_tiled_conf = {output41_conf.h/h_num_tiles, output41_conf.w/w_num_tiles, output41_conf.c};


		float *input41 = output34;

		float *output41 = (float *)mkl_calloc(output41_conf.h * output41_conf.w *
			output41_conf.c, bytes, alignment);

		float *input41_tiled = (float *)mkl_calloc(input41_tiled_conf.h * input41_tiled_conf.w *
			input41_tiled_conf.c, bytes, alignment);

		float *output41_tiled = (float *)mkl_calloc(output41_tiled_conf.h * output41_tiled_conf.w *
			output41_tiled_conf.c, bytes, alignment);


		//Conv42
		Conv_conf conv42_conf = {3, 3, 1, 1};
		Data_conf input42_conf = {28, 28, 512};
		Data_conf output42_conf = {28, 28, 512};


		Conv_conf conv42_tiled_conf = {3, 3, 1, 0};
		Data_conf input42_tiled_conf = {input42_conf.h/h_num_tiles + (conv42_conf.h - 1),
						input42_conf.w/w_num_tiles  + (conv42_conf.w - 1), input42_conf.c};
		Data_conf output42_tiled_conf = {output42_conf.h/h_num_tiles, output42_conf.w/w_num_tiles, output42_conf.c};


		float *input42 = output41;

		float *output42 = (float *)mkl_calloc(output42_conf.h * output42_conf.w *
			output42_conf.c, bytes, alignment);

		float *input42_tiled = (float *)mkl_calloc(input42_tiled_conf.h * input42_tiled_conf.w *
			input42_tiled_conf.c, bytes, alignment);

		float *output42_tiled = (float *)mkl_calloc(output42_tiled_conf.h * output42_tiled_conf.w *
			output42_tiled_conf.c, bytes, alignment);


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

		float *conv31_weights;
		float *conv31_biases;

		float *conv32_weights;
		float *conv32_biases;

		float *conv33_weights;
		float *conv33_biases;

		float *conv41_weights;
		float *conv41_biases;
		
		float *conv42_weights;
		float *conv42_biases;

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

		cnpy::NpyArray arr31 = cnpy::npy_load(weight_dir+"conv3_1_W.npy");
		conv31_weights = arr31.data<float>();

	    cnpy::NpyArray arr31_biases = cnpy::npy_load(weight_dir+"conv3_1_b.npy");
		conv31_biases = arr31_biases.data<float>();

		cnpy::NpyArray arr32 = cnpy::npy_load(weight_dir+"conv3_2_W.npy");
		conv32_weights = arr32.data<float>();

	    cnpy::NpyArray arr32_biases = cnpy::npy_load(weight_dir+"conv3_2_b.npy");
		conv32_biases = arr32_biases.data<float>();

		cnpy::NpyArray arr33 = cnpy::npy_load(weight_dir+"conv3_3_W.npy");
		conv33_weights = arr33.data<float>();

	    cnpy::NpyArray arr33_biases = cnpy::npy_load(weight_dir+"conv3_3_b.npy");
		conv33_biases = arr33_biases.data<float>();

		cnpy::NpyArray arr41 = cnpy::npy_load(weight_dir+"conv4_1_W.npy");
		conv41_weights = arr41.data<float>();

	    cnpy::NpyArray arr41_biases = cnpy::npy_load(weight_dir+"conv4_1_b.npy");
		conv41_biases = arr41_biases.data<float>();

		cnpy::NpyArray arr42 = cnpy::npy_load(weight_dir+"conv4_2_W.npy");
		conv42_weights = arr42.data<float>();

	    cnpy::NpyArray arr42_biases = cnpy::npy_load(weight_dir+"conv4_2_b.npy");
		conv42_biases = arr42_biases.data<float>();

	int times = 10;

	bool tiled = true;

	double tot_time = 0.0;

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
		pool_forward(input23, output23, input23_conf, output23_conf, pool2_conf);
		

		for (int h_tile = 0; h_tile < h_num_tiles; h_tile++) {
			for (int w_tile = 0; w_tile < w_num_tiles; w_tile++) {

				int h_base = h_tile * (input31_tiled_conf.h - (conv31_conf.h - 1));
				int w_base = w_tile * (input31_tiled_conf.w - (conv31_conf.w - 1));

				TILE_BASE tile_base = {h_base, w_base};

				load_tile(input31, input31_conf, tile_base, h_num_tiles, 
							input31_tiled, input31_tiled_conf);

				conv_im2row(input31_tiled, output31_tiled, conv31_weights, conv31_biases, conv31_tiled_conf,
					input31_tiled_conf, output31_tiled_conf);

				save_tile(output31_tiled, output31_tiled_conf, tile_base, output31, output31_conf);

			}
		}


		for (int h_tile = 0; h_tile < h_num_tiles; h_tile++) {
			for (int w_tile = 0; w_tile < w_num_tiles; w_tile++) {

				int h_base = h_tile * (input32_tiled_conf.h - (conv32_conf.h - 1));
				int w_base = w_tile * (input32_tiled_conf.w - (conv32_conf.w - 1));

				TILE_BASE tile_base = {h_base, w_base};

				load_tile(input32, input32_conf, tile_base, h_num_tiles, 
							input32_tiled, input32_tiled_conf);

				conv_im2row(input32_tiled, output32_tiled, conv32_weights, conv32_biases, conv32_tiled_conf,
					input32_tiled_conf, output32_tiled_conf);

				save_tile(output32_tiled, output32_tiled_conf, tile_base, output32, output32_conf);

			}
		}

		for (int h_tile = 0; h_tile < h_num_tiles; h_tile++) {
			for (int w_tile = 0; w_tile < w_num_tiles; w_tile++) {

				int h_base = h_tile * (input33_tiled_conf.h - (conv33_conf.h - 1));
				int w_base = w_tile * (input33_tiled_conf.w - (conv33_conf.w - 1));

				TILE_BASE tile_base = {h_base, w_base};

				load_tile(input33, input33_conf, tile_base, h_num_tiles, 
							input33_tiled, input33_tiled_conf);

				conv_im2row(input33_tiled, output33_tiled, conv33_weights, conv33_biases, conv33_tiled_conf,
					input33_tiled_conf, output33_tiled_conf);

				save_tile(output33_tiled, output33_tiled_conf, tile_base, output33, output33_conf);

			}
		}

		pool_forward(input34, output34, input34_conf, output34_conf, pool2_conf);

		for (int h_tile = 0; h_tile < h_num_tiles; h_tile++) {
			for (int w_tile = 0; w_tile < w_num_tiles; w_tile++) {

				int h_base = h_tile * (input41_tiled_conf.h - (conv41_conf.h - 1));
				int w_base = w_tile * (input41_tiled_conf.w - (conv41_conf.w - 1));

				TILE_BASE tile_base = {h_base, w_base};

				load_tile(input41, input41_conf, tile_base, h_num_tiles, 
							input41_tiled, input41_tiled_conf);

				conv_im2row(input41_tiled, output41_tiled, conv41_weights, conv41_biases, conv41_tiled_conf,
					input41_tiled_conf, output41_tiled_conf);

				save_tile(output41_tiled, output41_tiled_conf, tile_base, output41, output41_conf);

			}
		}
		for (int i = 0; i < times; i++) {
			start = std::chrono::system_clock::now();
			
			for (int h_tile = 0; h_tile < h_num_tiles; h_tile++) {
				for (int w_tile = 0; w_tile < w_num_tiles; w_tile++) {

					int h_base = h_tile * (input42_tiled_conf.h - (conv42_conf.h - 1));
					int w_base = w_tile * (input42_tiled_conf.w - (conv42_conf.w - 1));

					TILE_BASE tile_base = {h_base, w_base};

					load_tile(input42, input42_conf, tile_base, h_num_tiles, 
								input42_tiled, input42_tiled_conf);

					conv_im2row(input42_tiled, output42_tiled, conv42_weights, conv42_biases, conv42_tiled_conf,
						input42_tiled_conf, output42_tiled_conf);

					save_tile(output42_tiled, output42_tiled_conf, tile_base, output42, output42_conf);

				}
			}
			end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_time = end-start;
			tot_time += elapsed_time.count();
			// cout<<elapsed_time.count()<<endl;
		}
		// cout<<tot_time/times<<endl;
	}
	else {
		conv_im2row(input11, output11, conv11_weights, conv11_biases, conv11_conf,
				input11_conf, output11_conf);
		
		
		conv_im2row(input12, output12, conv12_weights, conv12_biases, conv12_conf,
				input12_conf, output12_conf);
		
		pool_forward(input13, output13, input13_conf, output13_conf,pool1_conf);
		
		conv_im2row(input21, output21, conv21_weights,conv21_biases, conv21_conf, input21_conf, output21_conf);
		
		conv_im2row(input22, output22, conv22_weights,conv22_biases, conv22_conf, input22_conf, output22_conf);
		
		pool_forward(input23, output23, input23_conf, output23_conf, pool2_conf);

		conv_im2row(input31, output31, conv31_weights,conv31_biases, conv31_conf, input31_conf, output31_conf);
		
		conv_im2row(input32, output32, conv32_weights,conv32_biases, conv32_conf, input32_conf, output32_conf);

		conv_im2row(input33, output33, conv33_weights,conv33_biases, conv33_conf, input33_conf, output33_conf);

		pool_forward(input34, output34, input34_conf, output34_conf, pool2_conf);

		conv_im2row(input41, output41, conv41_weights,conv41_biases, conv41_conf, input41_conf, output41_conf);
		
		for (int i = 0; i < times; i++) {
			
			for (int idx = 0; idx < output42_conf.h * output42_conf.w * output42_conf.c; idx++) {
				output42[idx] = 0.0f;
			}
			start = std::chrono::system_clock::now();
			conv_im2row(input42, output42, conv42_weights,conv42_biases, conv42_conf, input42_conf, output42_conf);
			end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_time = end-start;
			tot_time += elapsed_time.count();
			// cout<<elapsed_time.count()<<endl;
		}
		// cout<<tot_time/times<<endl;
	}

	cout<<tot_time/times<<endl;

	// std::chrono::duration<double> elapsed_time = end-start;

	// cout<<elapsed_time.count()<<endl;

	// for (int i = 0; i < output42_conf.h ; i++) {
	// 	for (int j = 0; j < output42_conf.w; j++) {
	// 		for (int k = 0; k < output42_conf.c; k++) {
	// 			int idx = (i * output42_conf.w + j) * output42_conf.c + k;
	// 			cout<<fixed<<setprecision(10)<<output42[idx]<<endl;
	// 		}
	// 	}
	// }
}