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

enum Layer_type {
	conv,
	pool,
	relu,
	fc,
	no_imp
};

void conv_forward(float *in, float *out, float *filter, Conv_conf conv_conf, Data_conf input_conf, Data_conf output_conf);
void conv_im2col(float *in, float *out, float *weights, float *biases, Conv_conf conv_conf,
					Data_conf input_conf, Data_conf output_conf);
void conv_im2row(float *in, float *out, float *weights, float *biases, Conv_conf conv_conf,
					Data_conf input_conf, Data_conf output_conf);
void pool_forward(float *in, float *out, Data_conf input_conf, Data_conf output_conf, Pool_conf pool_conf);
void relu_forward(float *in, float *out, Data_conf input_conf);
void fc_forward(float *in, float *out, float *weights, float *biases, int input_size, int output_size);
void fc_softmax_forward(float *in, float *out, float *weights, float *biases, int input_size, int output_size);
void softmax_forward(float *in, float *out, int input_size, int output_size);
float *patch_ret(float *in, float *out, float *weights, float *biases, Conv_conf conv_conf,
					Data_conf input_conf, Data_conf output_conf);
// using namespace std;
using namespace std::chrono;

#endif