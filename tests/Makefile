CFLAGS = -std=c++11
INCLUDE_PATH = -I../include
FLAGS = -lcnn -fopenmp -lz
INTEL_FLAGS = -mkl
OPENCV_FLAGS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

vgg_16_im2col_opt:
	icpc -I../include $(CFLAGS) vgg_16_im2col_opt.cpp -o vgg_16_im2col_opt.exec $(INTEL_FLAGS) $(FLAGS)
vgg16_org:
	icpc -I../include $(CFLAGS) vgg16_orig.cpp -o vgg16_orig.exec $(INTEL_FLAGS) $(FLAGS) $(OPENCV_FLAGS)
test_2layer_224:
	icpc $(INCLUDE_PATH) $(CFLAGS) test_2layer_224.cpp -o test_2layer_224.exec $(INTEL_FLAGS) $(FLAGS) $(OPENCV_FLAGS)
tiling_test:
	icpc $(INCLUDE_PATH) $(CFLAGS) tiling_test.cpp -o tiling_test.exec $(INTEL_FLAGS) $(FLAGS)