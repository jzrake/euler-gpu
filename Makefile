CU_FLAGS = -Xptxas -O3

all: euler1d sr1d

euler1d: euler1d.cu
	nvcc $(CU_FLAGS) -o $@ $^

sr1d: sr1d.cu
	nvcc $(CU_FLAGS) -o $@ $^
