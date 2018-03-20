//opencv headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <input.hpp>
#include <iostream>

using namespace cv;

//read and put the image in data
int read_image_rgb(std::string filename, Image_cfg cfg, float *data) {
	Mat img, mat = Mat::zeros(cfg.rows, cfg.cols, CV_8UC3);

  img = imread(filename, CV_LOAD_IMAGE_COLOR);
  cvtColor(img, img, CV_RGB2BGR);
  if(! img.data)                             // Check for invalid input
  {
    std::cout <<  "Could not open or find the image" << std::endl ;
    return -1;
  }

  resize(img, mat, Size(cfg.rows, cfg.cols), 0, 0, INTER_LINEAR);

	std::vector<uchar> array;
	if (mat.isContinuous()) {
  		array.assign(mat.datastart, mat.dataend);
	}
	else {
		for (int i = 0; i < mat.rows; ++i) {
  		array.insert(array.end(), mat.ptr<uchar>(i), mat.ptr<uchar>(i)+mat.cols);
		}
	}

	for (int i = 0; i < array.size(); i++) {
		data[i] = (float)array[i];
	}

	return 0;
}