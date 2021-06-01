CU_FLAGS = -Xptxas -O3
EXE = euler2d

all: euler2d

euler2d: euler2d.cu
	nvcc $(CU_FLAGS) -o $@ $^

clean:
	$(RM) $(EXE)
