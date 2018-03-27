#include <tiling.hpp>

std::vector<std::vector<Data_conf>> create_tiled_conf(Data_conf input_conf, std::vector<Layer_conf> layers, 
								int num_tiles) {

	//calculate output sizes of layers
	std::vector<std::vector<Data_conf>> ret;

	std::vector<Data_conf> ouput;
	Data_conf curr_input = input_conf;

	int num_layers = layers.size();
	for (int i = 0; i < layers.size(); i++) {
		Layer_conf curr_layer = layers[i];
		// std::cout<<"it"<<std::endl;
		if (curr_layer.type == conv || curr_layer.type == pool) {
			int out_h = (curr_input.h + 2*curr_layer.p - curr_layer.h)/curr_layer.s + 1;
			int out_w = (curr_input.w + 2*curr_layer.p - curr_layer.w)/curr_layer.s + 1;
			Data_conf curr_output = {out_h, out_w, curr_layer.c};
			output.push_back(curr_output);
			std::cout<<"["<<curr_output.h<<"x"<<curr_output.w<<"x"
									<<curr_output.c<<"]"<<std::endl;
		}
	}

	
	// std::cout<<"sdfsd"<<std::endl;
	return ret;
}