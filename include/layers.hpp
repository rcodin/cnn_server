#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <chrono>

struct Conv_conf {
	int h;
	int w;
	int stride;
	int pad;
};

struct Pool_conf {
	int h;
	int w;
	int stride;
};

struct Data_conf {
	int h;
	int w;
	int c;
};

/*
 **conv   = convolutional layer
 **pool   = pooling layer
 **relu   = Rectified linear unit
 **fc 	  = fully connected or dense layer
 **no_imp = not implemented in the library
*/

typedef enum Layer_type {
	conv,
	pool,
	relu,
	fc,
	no_imp
} ltype;

/*
 **type = type of the layer
 **h = height
 **w = weight
 **c = channels
 **s = stride
 **p = pad
 *** In case of fc layer the h represents the output size
*/

struct Layer_conf {
	ltype type;
	int h;
	int w;
	int c;
	int s;
	int p;
};


struct TILE_Conf {
	int h_s;
	int h_e;
	int w_s;
	int w_e;
	int c;
};

void conv_forward(float *in, float *out, float *filter, Conv_conf conv_conf, Data_conf input_conf, Data_conf output_conf);
void conv_im2col(float *in, float *out, float *weights, float *biases, Conv_conf conv_conf,
					Data_conf input_conf, Data_conf output_conf);
void conv_im2row(float *in, float *out, float *weights, float *biases, Conv_conf conv_conf,
					Data_conf input_conf, Data_conf output_conf);
void conv_im2row_mod(float *in, float *out, float *weights, float *biases, Conv_conf conv_conf,
					Data_conf input_conf, Data_conf output_conf);

void pool_forward(float *in, float *out, Data_conf input_conf, Data_conf output_conf, Pool_conf pool_conf);
void relu_forward(float *in, float *out, Data_conf input_conf);
void fc_forward(float *in, float *out, float *weights, float *biases, int input_size, int output_size);
void fc_softmax_forward(float *in, float *out, float *weights, float *biases, int input_size, int output_size);
void softmax_forward(float *in, float *out, int input_size, int output_size);
float *patch_ret(float *in, float *out, float *weights, float *biases, Conv_conf conv_conf,
					Data_conf input_conf, Data_conf output_conf);

void conv_forward_bias(float *in, float *out, float *weights, float *biases, Conv_conf conv_conf,
									Data_conf input_conf, Data_conf output_conf);
// using namespace std;
using namespace std::chrono;

#endif