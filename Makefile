CUFLAGS = -Xptxas -O3
CFLAGS = -Wall -O3 -fopenmp

EXE = div2d

all: $(EXE)

div2d: div2d.cu
	nvcc $(CUFLAGS) -o $@ $<

clean:
	$(RM) $(EXE)
