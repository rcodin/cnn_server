#ifndef TILING_HPP
#define TILING_HPP

#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <layers.hpp>

struct TILE_IDX {
	int h;
	int w;
};

struct TILE_BASE {
	int h;
	int w;
};

struct PAD_CONF {
	int h_s;
	int h_e;
	int w_s;
	int w_e;
};

std::vector<std::vector<Data_conf>> create_tiled_conf(Data_conf input_conf, std::vector<Layer_conf> layers, 
								int num_tiles);

#endif