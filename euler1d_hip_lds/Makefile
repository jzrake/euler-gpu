HIP_FLAGS = -O3
EXE = euler1d

all: euler1d

euler1d: euler1d.cpp
	hipcc $(HIP_FLAGS) -o $@ $^

clean:
	$(RM) $(EXE)
