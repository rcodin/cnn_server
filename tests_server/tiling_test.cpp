#include <tiling.hpp>
#include <layers.hpp>
#include <iostream>


using namespace std;

int main() {
	size_t bytes = sizeof(float);
	int alignment = bytes * 8;

	Data_conf input11_conf = {224, 224, 3};
	Layer_conf l11_conf = {conv, 3, 3, 64, 1, 1};
	Layer_conf l12_conf = {conv, 3, 3, 64, 1, 1};
	Layer_conf l13_conf = {pool, 2, 2, 64, 2, 0};
	Layer_conf l21_conf = {conv, 3, 3, 256, 1, 1};

	vector<Layer_conf> layers_g1;

	layers_g1.push_back(l11_conf);
	layers_g1.push_back(l12_conf);
	layers_g1.push_back(l13_conf);
	layers_g1.push_back(l21_conf);

	create_tiled_conf(input11_conf, layers_g1, 8);
}