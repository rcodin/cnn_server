// #include <tiling.hpp>

// std::vector<std::vector<Data_conf>> create_tiled_conf(Data_conf input_conf, std::vector<Layer_conf> layers, 
// 								int num_tiles) {

// 	//calculate output sizes of layers
// 	std::vector<std::vector<Data_conf>> ret (num_tiles * num_tiles);

// 	std::vector<Data_conf> ouput;
// 	Data_conf curr_input = input_conf;

// 	int num_layers = layers.size();
// 	for (int i = 0; i < layers.size(); i++) {
// 		Layer_conf curr_layer = layers[i];
// 		// std::cout<<"it"<<std::endl;
// 		if (curr_layer.type == conv || curr_layer.type == pool) {
// 			int out_h = (curr_input.h + 2*curr_layer.p - curr_layer.h)/curr_layer.s + 1;
// 			int out_w = (curr_input.w + 2*curr_layer.p - curr_layer.w)/curr_layer.s + 1;
// 			Data_conf curr_output = {out_h, out_w, curr_layer.c};
// 			output.push_back(curr_output);
// 			std::cout<<"["<<curr_output.h<<"x"<<curr_output.w<<"x"
// 									<<curr_output.c<<"]"<<std::endl;
// 		}
// 	}

// 	for (int h_idx = 0; h_idx < num_tiles; h_idx++) {
// 		for (int w_idx = 0; w_idx < num_tiles; w_idx++) {

// 			int tile_idx = h_idx * num_tiles + w_idx;

// 			Data_conf curr_conf = output[output.size() - 1];
// 			Data_conf curr_tiled_conf = {curr_conf.h/num_tiles, curr_conf.w/num_tiles, curr_conf.c};

// 			std::vector<TILE_Conf> w_tile_conf (layers.size() + 1);


// 			for (std::vector<Layer_conf>::reverse_iterator rit = layers.rbegin();
// 							rit != layers.rend(); rit++) {

// 				// w_tile_conf.insert(w_tile_conf.begin(), curr_tiled_conf);
				
// 				Layer_conf curr_layer = *rit;

// 				int h_size = (curr_tiled_conf.h - 1)*curr_layer.s + curr_layer.h;
// 				int w_size = (curr_tiled_conf.w - 1)*curr_layer.s + curr_layer.w;
// 				int c_size = curr_tiled_conf.c;
				
// 				int h_s, h_e;
// 				int h_s, h_e;

// 				if (h_idx == 0) {
// 					h_size -= curr_layer.p;
// 					h_s = 0;
// 					h_e = h_size - 1;
// 				}
// 				else if (h_idx == (num_tiles - 1)) {
// 					TILE_Conf last_tile = ret[tile_idx - 1];
// 					h_size -= curr_layer.p;
// 					h_s = last_tile.h_e - (curr_layer.h - curr_layer.s)/2;
// 					h_e = h_s + h_size - 1;
// 				}
// 				else {
// 					TILE_Conf last_tile = ret[tile_idx - 1];
// 					h_s = last_tile.h_e - (curr_layer.h - curr_layer.s)/2;
// 					h_e = h_s + h_size - 1;
// 				}

// 				if (w_idx == 0) {
// 					w_size -= curr_layer.p;
// 					w_s = 0;
// 					w_e = w_size - 1;
// 				}
// 				else if (w_idx == (num_tiles - 1)) {
// 					TILE_Conf last_tile = ret[tile_idx - 1];
// 					w_size -= curr_layer.p;
// 					w_s = last_tile.w_e - (curr_layer.w - curr_layer.s)/2;
// 					w_e = w_s + w_size - 1;
// 				}
// 				else {
// 					TILE_Conf last_tile = ret[tile_idx - 1];
// 					w_s = last_tile.w_e - (curr_layer.w - curr_layer.s)/2;
// 					w_e = w_s + w_size - 1;
// 				}

// 				TILE_Conf tile_conf = {h_s, h_e, w_s, w_e, curr_layer.c};
// 				curr_tiled_conf = {h_size, w_size, };
// 			}
// 		}
// 	}
// 	// std::cout<<"sdfsd"<<std::endl;
// 	return ret;
// }