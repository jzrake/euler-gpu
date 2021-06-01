CU_FLAGS = -Xptxas -O3
EXE = euler3d

all: euler3d

euler3d: euler3d.cu
	nvcc $(CU_FLAGS) -o $@ $^

clean:
	$(RM) $(EXE)
