CUFLAGS = -Xptxas -O3
CFLAGS = -Wall -O3

EXE = euler1d_gpu euler1d_cpu

all: $(EXE)

euler1d_gpu: euler1d_gpu.cu
	nvcc $(CUFLAGS) -o $@ $^

euler1d_cpu: euler1d_cpu.c
	cc $(CFLAGS) -o $@ $^ -lm

clean:
	$(RM) $(EXE)
