CUFLAGS = -Xptxas -O3
CFLAGS = -Wall -O3 -fopenmp

EXE = euler1d_cpu euler1d_gpu

all: $(EXE)

euler1d_gpu: euler1d_gpu.cu euler1d.c
	nvcc $(CUFLAGS) -o $@ $<

euler1d_cpu: euler1d_cpu.c euler1d.c
	cc $(CFLAGS) -o $@ $< -lm

clean:
	$(RM) $(EXE)
