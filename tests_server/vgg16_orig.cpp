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

		//Conv21
		Conv_conf conv21_conf = {3, 3, 1, 1};
		Data_conf input21_conf = {112, 112, 64};
		Data_conf output21_conf = {112, 112, 128};

		//Conv22
		Conv_conf conv22_conf = {3, 3, 1, 1};
		Data_conf input22_conf = {112, 112, 128};
		Data_conf output22_conf = {112, 112, 128};

		//Pool2	
		Pool_conf pool2_conf = {2, 2};
		Data_conf input23_conf = {112, 112, 128};
		Data_conf output23_conf = {56, 56, 128};


		//Conv31
		Conv_conf conv31_conf = {3, 3, 1, 1};
		Data_conf input31_conf = {56, 56, 128};
		Data_conf output31_conf = {56, 56, 256};

		//Conv32
		Conv_conf conv32_conf = {3, 3, 1, 1};
		Data_conf input32_conf = {56, 56, 256};
		Data_conf output32_conf = {56, 56, 256};

		//Conv33
		Conv_conf conv33_conf = {3, 3, 1, 1};
		Data_conf input33_conf = {56, 56, 256};
		Data_conf output33_conf = {56, 56, 256};

		//Pool3
		Pool_conf pool3_conf = {2, 2};
		Data_conf input34_conf = {56, 56, 256};
		Data_conf output34_conf = {28, 28, 256};

		//Conv41
		Conv_conf conv41_conf = {3, 3, 1, 1};
		Data_conf input41_conf = {28, 28, 256};
		Data_conf output41_conf = {28, 28, 512};

		//Conv42
		Conv_conf conv42_conf = {3, 3, 1, 1};
		Data_conf input42_conf = {28, 28, 512};
		Data_conf output42_conf = {28, 28, 512};

		//Conv43
		Conv_conf conv43_conf = {3, 3, 1, 1};
		Data_conf input43_conf = {28, 28, 512};
		Data_conf output43_conf = {28, 28, 512};

		//Pool4
		Pool_conf pool4_conf = {2, 2};
		Data_conf input44_conf = {28, 28, 512};
		Data_conf output44_conf = {14, 14, 512};

		//Conv51
		Conv_conf conv51_conf = {3, 3, 1, 1};
		Data_conf input51_conf = {14, 14, 512};
		Data_conf output51_conf = {14, 14, 512};

		//Conv52
		Conv_conf conv52_conf = {3, 3, 1, 1};
		Data_conf input52_conf = {14, 14, 512};
		Data_conf output52_conf = {14, 14, 512};

		//Conv53
		Conv_conf conv53_conf = {3, 3, 1, 1};
		Data_conf input53_conf = {14, 14, 512};
		Data_conf output53_conf = {14, 14, 512};

		//Pool5
		Pool_conf pool5_conf = {2, 2};
		Data_conf input54_conf = {14, 14, 512};
		Data_conf output54_conf = {7, 7, 512};	

		//fc1 flattening
		Data_conf input6_conf = {7, 7, 512};
		int output6_conf = 4096;
		
		//fc2
		int input7_conf = 4096;
		int output7_conf = 4096;
		
		//fc3_softmax
		int input8_conf = 4096;
		int output8_conf = 1000;

	float *input11 = (float *)mkl_calloc(input11_conf.h * input11_conf.w * input11_conf.c , bytes, alignment);
	float *output11 = (float *)mkl_calloc(output11_conf.h * output11_conf.w * output11_conf.c , bytes, alignment);
	float *output12 = (float *)mkl_calloc(output12_conf.h * output12_conf.w * output12_conf.c , bytes, alignment);
	float *output13 = (float *)mkl_calloc(output13_conf.h * output13_conf.w * output13_conf.c , bytes, alignment);

	float *output21 = (float *)mkl_calloc(output21_conf.h * output21_conf.w * output21_conf.c , bytes, alignment);
	float *output22 = (float *)mkl_calloc(output22_conf.h * output22_conf.w * output22_conf.c , bytes, alignment);
	float *output23 = (float *)mkl_calloc(output23_conf.h * output23_conf.w * output23_conf.c , bytes, alignment);

	float *output31 = (float *)mkl_calloc(output31_conf.h * output31_conf.w * output31_conf.c , bytes, alignment);
	float *output32 = (float *)mkl_calloc(output32_conf.h * output32_conf.w * output32_conf.c , bytes, alignment);
	float *output33 = (float *)mkl_calloc(output33_conf.h * output33_conf.w * output33_conf.c , bytes, alignment);
	float *output34 = (float *)mkl_calloc(output34_conf.h * output34_conf.w * output34_conf.c , bytes, alignment);

	float *output41 = (float *)mkl_calloc(output41_conf.h * output41_conf.w * output41_conf.c , bytes, alignment);
	float *output42 = (float *)mkl_calloc(output42_conf.h * output42_conf.w * output42_conf.c , bytes, alignment);
	float *output43 = (float *)mkl_calloc(output43_conf.h * output43_conf.w * output43_conf.c , bytes, alignment);
	float *output44 = (float *)mkl_calloc(output44_conf.h * output44_conf.w * output44_conf.c , bytes, alignment);

	float *output51 = (float *)mkl_calloc(output51_conf.h * output51_conf.w * output51_conf.c , bytes, alignment);
	float *output52 = (float *)mkl_calloc(output52_conf.h * output52_conf.w * output52_conf.c , bytes, alignment);
	float *output53 = (float *)mkl_calloc(output53_conf.h * output53_conf.w * output53_conf.c , bytes, alignment);
	float *output54 = (float *)mkl_calloc(output54_conf.h * output54_conf.w * output54_conf.c , bytes, alignment);

	float *output6 = (float *)mkl_calloc(output6_conf , bytes, alignment);
	float *output7 = (float *)mkl_calloc(output7_conf , bytes, alignment);
	float *output8 = (float *)mkl_calloc(output8_conf , bytes, alignment);


	//allocating filers
	float *conv11_weights; //= (float *)mkl_malloc(output11_conf.c * conv11_conf.h * conv11_conf.w * input11_conf.c * bytes,  alignment);
	float *conv12_weights; //= (float *)mkl_malloc(output12_conf.c * conv12_conf.h * conv12_conf.w * input12_conf.c * bytes,  alignment);

	float *conv21_weights; //= (float *)mkl_malloc(output21_conf.c * conv21_conf.h * conv21_conf.w * input21_conf.c * bytes,  alignment);
	float *conv22_weights; //= (float *)mkl_malloc(output22_conf.c * conv22_conf.h * conv22_conf.w * input22_conf.c * bytes,  alignment);

	float *conv31_weights; //= (float *)mkl_malloc(output31_conf.c * conv31_conf.h * conv31_conf.w * input31_conf.c * bytes,  alignment);
	float *conv32_weights; //= (float *)mkl_malloc(output32_conf.c * conv32_conf.h * conv32_conf.w * input32_conf.c * bytes,  alignment);
	float *conv33_weights; //= (float *)mkl_malloc(output33_conf.c * conv32_conf.h * conv32_conf.w * input33_conf.c * bytes,  alignment);

	float *conv41_weights; //= (float *)mkl_malloc(output41_conf.c * conv41_conf.h * conv41_conf.w * input41_conf.c * bytes,  alignment);
	float *conv42_weights; //= (float *)mkl_malloc(output42_conf.c * conv42_conf.h * conv42_conf.w * input42_conf.c * bytes,  alignment);
	float *conv43_weights; //= (float *)mkl_malloc(output43_conf.c * conv43_conf.h * conv43_conf.w * input43_conf.c * bytes,  alignment);

	float *conv51_weights; //= (float *)mkl_malloc(output51_conf.c * conv51_conf.h * conv51_conf.w * input51_conf.c * bytes,  alignment);
	float *conv52_weights; //= (float *)mkl_malloc(output52_conf.c * conv52_conf.h * conv52_conf.w * input52_conf.c * bytes,  alignment);
	float *conv53_weights; //= (float *)mkl_malloc(output53_conf.c * conv53_conf.h * conv53_conf.w * input53_conf.c * bytes,  alignment);

	//allocating biases

	float *conv11_biases; //= (float *)mkl_malloc(output11_conf.c  * bytes,  alignment);
	float *conv12_biases; //= (float *)mkl_malloc(output12_conf.c  * bytes,  alignment);

	float *conv21_biases; //= (float *)mkl_malloc(output21_conf.c  * bytes,  alignment);
	float *conv22_biases; //= (float *)mkl_malloc(output22_conf.c  * bytes,  alignment);

	float *conv31_biases; //= (float *)mkl_malloc(output31_conf.c  * bytes,  alignment);
	float *conv32_biases; //= (float *)mkl_malloc(output32_conf.c  * bytes,  alignment);
	float *conv33_biases; //= (float *)mkl_malloc(output33_conf.c  * bytes,  alignment);

	float *conv41_biases; //= (float *)mkl_malloc(output41_conf.c  * bytes,  alignment);
	float *conv42_biases; //= (float *)mkl_malloc(output42_conf.c  * bytes,  alignment);
	float *conv43_biases; //= (float *)mkl_malloc(output43_conf.c  * bytes,  alignment);

	float *conv51_biases; //= (float *)mkl_malloc(output51_conf.c  * bytes,  alignment);
	float *conv52_biases; //= (float *)mkl_malloc(output52_conf.c  * bytes,  alignment);
	float *conv53_biases; //= (float *)mkl_malloc(output53_conf.c  * bytes,  alignment);

	//load fc weights into a new array
	float *fc1_weights;// = (float *)mkl_malloc(input6_conf.h * input6_conf.w * input6_conf.c * output6_conf * bytes, alignment);
	float *fc2_weights;// = (float *)mkl_malloc(input7_conf * output7_conf * bytes, alignment);
	float *fc3_weights;// = (float *)mkl_malloc(input8_conf * output8_conf * bytes, alignment);
	

	//load fc biases into a new array
	float *fc1_biases;
	float *fc2_biases;
	float *fc3_biases;

	//load image

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

    if (err) {
    	std::cerr<<"Error: unable to read file"<<std::endl;
    	return -1;
    }

		//load conv weights
	    cnpy::NpyArray arr11 = cnpy::npy_load(weight_dir+"conv1_1_W.npy");
		conv11_weights = arr11.data<float>();

		cnpy::NpyArray arr12 = cnpy::npy_load(weight_dir+"conv1_2_W.npy");
		conv12_weights = arr12.data<float>();

		cnpy::NpyArray arr21 = cnpy::npy_load(weight_dir+"conv2_1_W.npy");
		conv21_weights = arr21.data<float>();

		// float *temp;
		cnpy::NpyArray arr22 = cnpy::npy_load(weight_dir+"conv2_2_W.npy");
		conv22_weights = arr22.data<float>();

		// std::cout<<conv22_weights[(output21_conf.c * conv21_conf.h * conv21_conf.w * input21_conf.c) - 1]<<std::endl;

		cnpy::NpyArray arr31 = cnpy::npy_load(weight_dir+"conv3_1_W.npy");
		conv31_weights = arr31.data<float>();

		cnpy::NpyArray arr32 = cnpy::npy_load(weight_dir+"conv3_2_W.npy");
		conv32_weights = arr32.data<float>();

		cnpy::NpyArray arr33 = cnpy::npy_load(weight_dir+"conv3_3_W.npy");
		conv33_weights = arr33.data<float>();

		cnpy::NpyArray arr41 = cnpy::npy_load(weight_dir+"conv4_1_W.npy");
		conv41_weights = arr41.data<float>();

		cnpy::NpyArray arr42 = cnpy::npy_load(weight_dir+"conv4_2_W.npy");
		conv42_weights = arr42.data<float>();

		cnpy::NpyArray arr43 = cnpy::npy_load(weight_dir+"conv4_3_W.npy");
		conv43_weights = arr43.data<float>();

		cnpy::NpyArray arr51 = cnpy::npy_load(weight_dir+"conv5_1_W.npy");
		conv51_weights = arr51.data<float>();

		cnpy::NpyArray arr52 = cnpy::npy_load(weight_dir+"conv5_2_W.npy");
		conv52_weights = arr52.data<float>();

		cnpy::NpyArray arr53 = cnpy::npy_load(weight_dir+"conv5_3_W.npy");
		conv53_weights = arr53.data<float>();

		// load conv biases
	    cnpy::NpyArray arr11_biases = cnpy::npy_load(weight_dir+"conv1_1_b.npy");
		conv11_biases = arr11_biases.data<float>();

		cnpy::NpyArray arr12_biases = cnpy::npy_load(weight_dir+"conv1_2_b.npy");
		conv12_biases = arr12_biases.data<float>();

		cnpy::NpyArray arr21_biases = cnpy::npy_load(weight_dir+"conv2_1_b.npy");
		conv21_biases = arr21_biases.data<float>();

		cnpy::NpyArray arr22_biases = cnpy::npy_load(weight_dir+"conv2_2_b.npy");
		conv22_biases = arr22_biases.data<float>();

		cnpy::NpyArray arr31_biases = cnpy::npy_load(weight_dir+"conv3_1_b.npy");
		conv31_biases = arr31_biases.data<float>();

		cnpy::NpyArray arr32_biases = cnpy::npy_load(weight_dir+"conv3_2_b.npy");
		conv32_biases = arr32_biases.data<float>();

		cnpy::NpyArray arr33_biases = cnpy::npy_load(weight_dir+"conv3_3_b.npy");
		conv33_biases = arr33_biases.data<float>();

		cnpy::NpyArray arr41_biases = cnpy::npy_load(weight_dir+"conv4_1_b.npy");
		conv41_biases = arr41_biases.data<float>();

		cnpy::NpyArray arr42_biases = cnpy::npy_load(weight_dir+"conv4_2_b.npy");
		conv42_biases = arr42_biases.data<float>();

		cnpy::NpyArray arr43_biases = cnpy::npy_load(weight_dir+"conv4_3_b.npy");
		conv43_biases = arr43_biases.data<float>();

		cnpy::NpyArray arr51_biases = cnpy::npy_load(weight_dir+"conv5_1_b.npy");
		conv51_biases = arr51_biases.data<float>();

		cnpy::NpyArray arr52_biases = cnpy::npy_load(weight_dir+"conv5_2_b.npy");
		conv52_biases = arr52_biases.data<float>();

		cnpy::NpyArray arr53_biases = cnpy::npy_load(weight_dir+"conv5_3_b.npy");
		conv53_biases = arr53_biases.data<float>();


		//fc layer weights
		cnpy::NpyArray arr6_weights = cnpy::npy_load(weight_dir+"fc6_W.npy");
		fc1_weights = arr6_weights.data<float>();

		cnpy::NpyArray arr7_weights = cnpy::npy_load(weight_dir+"fc7_W.npy");
		fc2_weights = arr7_weights.data<float>();

		cnpy::NpyArray arr8_weights = cnpy::npy_load(weight_dir+"fc8_W.npy");
		fc3_weights = arr8_weights.data<float>();

		//fc layer biases
		cnpy::NpyArray arr6_biases = cnpy::npy_load(weight_dir+"fc6_b.npy");
		fc1_biases = arr6_biases.data<float>();

		cnpy::NpyArray arr7_biases = cnpy::npy_load(weight_dir+"fc7_b.npy");
		fc2_biases = arr7_biases.data<float>();

		cnpy::NpyArray arr8_biases = cnpy::npy_load(weight_dir+"fc8_b.npy");
		fc3_biases = arr8_biases.data<float>();

	//Group 1
	conv_im2row(input11, output11, conv11_weights,conv11_biases, conv11_conf, input11_conf, output11_conf);
	conv_im2row(output11, output12, conv12_weights, conv12_biases, conv12_conf, input12_conf, output12_conf);
	pool_forward(output12, output13, input13_conf, input21_conf,pool1_conf);

	//Group 2
	conv_im2row(output13, output21, conv21_weights,conv21_biases, conv21_conf, input21_conf, output21_conf);
	conv_im2row(output21, output22, conv22_weights,conv22_biases, conv22_conf, input22_conf, output22_conf);
	pool_forward(output22, output23, input23_conf, input31_conf, pool2_conf);

	// //Group 3
	conv_im2row(output23, output31, conv31_weights,conv31_biases, conv31_conf, input31_conf, output31_conf);
	conv_im2row(output31, output32, conv32_weights,conv32_biases, conv32_conf, input32_conf, output32_conf);
	conv_im2row(output32, output33, conv33_weights,conv33_biases, conv33_conf, input33_conf, output33_conf);
	pool_forward(output33, output34, input34_conf, input41_conf, pool3_conf);
	
	// Group 4
	conv_im2row(output34, output41, conv41_weights,conv41_biases, conv41_conf, input41_conf, output41_conf);
	conv_im2row(output41, output42, conv42_weights,conv42_biases, conv42_conf, input42_conf, output42_conf);
	conv_im2row(output42, output43, conv43_weights,conv43_biases, conv43_conf, input43_conf, output43_conf);
	pool_forward(output43, output44, input44_conf, input51_conf, pool4_conf);

	// for (int i = 0; i < 512; i++)

	// Group 5
	conv_im2row(output44, output51, conv51_weights,conv51_biases, conv51_conf, input51_conf, output51_conf);
	conv_im2row(output51, output52, conv52_weights,conv52_biases, conv52_conf, input52_conf, output52_conf);
	conv_im2row(output52, output53, conv53_weights,conv53_biases, conv53_conf, input53_conf, output53_conf);
	pool_forward(output53, output54, input54_conf, input6_conf,pool5_conf);
	
	// for (int i = 0; i < (output54_conf.h * output54_conf.w * output54_conf.c); i++)
	// 	std::cout<<std::fixed<<std::setprecision(10)<<output54[i]<<std::endl;

	// fc1
	fc_forward(output54, output6, fc1_weights, fc1_biases, input6_conf.h * input6_conf.w * input6_conf.c,
					output6_conf);
	//fc2
	fc_forward(output6, output7, fc2_weights, fc2_biases,input7_conf, output7_conf);
	

	//fc3
	fc_softmax_forward(output7, output8, fc3_weights, fc3_biases, input8_conf, output8_conf);

	// for (int i = 0; i < (output8_conf); i++)
	// 	std::cout<<std::fixed<<std::setprecision(10)<<output8[i]<<std::endl;
	// while(1);
	int idx = get_highest_prob(output8, output8_conf);

	// std::cout<<"Highest prob class is : "<<idx<<std::endl;

	return 0;
}