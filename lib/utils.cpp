#include <utils.hpp>
#include <layers.hpp>

void *alloc_1D(int i, size_t bytes) {
	void *ret = (void *)calloc(i, bytes);

	return ret;	
}


void *alloc_2D(int i, int j, size_t bytes) {
	void **ret = (void **)calloc(i, sizeof(void *));

	for (int i_idx = 0; i_idx < i; i_idx++)
		ret[i_idx] = calloc(j, bytes);
	return ret;	
}

void *alloc_3D(int i, int j, int k, size_t bytes) {
	void ***ret = (void ***)calloc(i, sizeof(void *));

	for (int i_idx = 0; i_idx < i; i_idx++) {
		ret[i_idx] = (void **)calloc(j, sizeof(void *));

		for (int j_idx = 0; j_idx < j; j_idx++)
			ret[i_idx][j_idx] = calloc(k, bytes);
	}

	// std::cout<<ret<<std::endl;
	if (ret == NULL)
		std::cout<<"Memory not allocated";
	return ret;
}

void *alloc_4D(int i, int j, int k, int l, size_t bytes) {
	void ****ret = (void ****)calloc(i, sizeof(void *));

	for (int i_idx = 0; i_idx < i; i_idx++) {
		ret[i_idx] = (void ***)calloc(j, sizeof(void *));

		for (int j_idx = 0; j_idx < j; j_idx++) {
			ret[i_idx][j_idx] = (void **)calloc(k, sizeof(void *));

			for (int k_idx = 0; k_idx < k; k_idx++)
				ret[i_idx][j_idx][k_idx] = calloc(l, bytes);
		}
	}

	// printf("%d\n", i * j * k * l * bytes);
	if (ret == NULL)
		std::cout<<"Memory not allocated";
	return ret;
}

void print_conf_cfg(Conv_conf cfg, Data_conf input_cfg, Data_conf output_cfg) {
	std::cout<<"Conv : ";
	std::cout<<"["<<output_cfg.c<<"x"<<cfg.h<<"x"<<cfg.w<<"x"<<input_cfg.c<<"]"<<std::endl;
	std::cout<<"Input : ";
	std::cout<<"["<<input_cfg.h<<"x"<<input_cfg.w<<"x"<<input_cfg.c<<std::endl;
}

void print_pool_cfg(Pool_conf pool_conf, Data_conf input_cfg) {
	std::cout<<"Pool : ";
	std::cout<<"["<<pool_conf.h<<"x"<<pool_conf.w<<"]"<<std::endl;
	std::cout<<"Input : ";
	std::cout<<"["<<input_cfg.h<<"x"<<input_cfg.w<<"x"<<input_cfg.c<<"]"<<std::endl;
	std::cout<<std::endl;
}

void print_linearize_conv_cfg(Data_conf input_cfg, int output_cfg) {
	std::cout<<"Input : "<<"["<<input_cfg.h<<"x"<<input_cfg.w<<"x"<<input_cfg.c<<"]"<<std::endl;
	std::cout<<"Output : "<<"["<<output_cfg<<"]"<<std::endl;
	std::cout<<std::endl;
}

void print_fc_cfg(int input_cfg, int output_cfg) {
	std::cout<<"Input : "<<"["<<input_cfg<<"]"<<std::endl;
	std::cout<<"Output : "<<"["<<output_cfg<<"]"<<std::endl;
	std::cout<<std::endl;
}

void free_mem(void *ptr) {
	free(ptr);
}
void replicate_across_cols(float *input, float *output, int rows, int cols) {
	for (int r = 0; r < rows; r++) {
		int val = input[r];
		for (int c = 0; c < cols; c++) {
			output[r * cols + c] += val;
		}
	}
}

void replicate_across_rows(float *input, float *output, int rows, int cols) {
	for (int r = 0; r < rows; r++) {
		// int val = input[r];
		for (int c = 0; c < cols; c++) {
			output[r * cols + c] += input[c];
		}
	}
}

float get_highest_prob(float *data, int data_size) {

	unsigned int max_idx;
	float max = -5.0f;
	for (int i = 0; i < data_size; i++) {
		if (max < data[i]) {
			max = data[i];
			max_idx = i;
		}
		// std::cout<<data[i]<<std::endl;
	}
	return max_idx;
}