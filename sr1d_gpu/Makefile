CU_FLAGS = -Xptxas -O3
EXE = sr1d

all: sr1d

sr1d: sr1d.cu
	nvcc $(CU_FLAGS) -o $@ $^

clean:
	$(RM) $(EXE)
