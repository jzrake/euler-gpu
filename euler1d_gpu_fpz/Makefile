CU_FLAGS = -Xptxas -O3
EXE = euler1d

all: euler1d

euler1d: euler1d.cu
	nvcc $(CU_FLAGS) -o $@ $^

clean:
	$(RM) $(EXE)
