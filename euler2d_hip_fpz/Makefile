CU_FLAGS = -O3
EXE = euler2d

all: euler2d

euler2d: euler2d.cpp
	hipcc $(CU_FLAGS) -o $@ $^

clean:
	$(RM) $(EXE)
