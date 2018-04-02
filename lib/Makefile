program_NAME := libcnn.so

SRCS = blas.cpp layers_forward_pass.cpp utils.cpp im2col.cpp cnpy.cpp input.cpp tiling.cpp
OBJS = blas.o layers_forward_pass.o utils.o im2col.o cnpy.o input.o tiling.o
INCLUDE_DIR := ../include
CFLAGS = -std=c++11 -fopenmp
MKL_FLAGS = -mkl 
OPENCV_FLAGS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

.PHONY: all

all: $(program_NAME)

$(program_NAME): $(SRCS) 
	icpc -fPIC -I$(INCLUDE_DIR) $(CFLAGS) -O -c -D TILE=1 $(SRCS) -fopenmp $(MKL_FLAGS) $(OPENCV_FLAGS)
	icpc -shared -o libcnn.so $(OBJS) -lm -fopenmp  $(MKL_FLAGS) $(OPENCV_FLAGS)