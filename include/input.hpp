#ifndef INPUT_HPP
#define INPUT_HPP

struct Input3D {
	int h;
	int w;
	int c;
	float ***data;	
};

struct Image_cfg {
	int rows;
	int cols;
};

int read_image_rgb(std::string filename, Image_cfg cfg, float *data);
#endif