CFLAGS = -Wall -O3 -lm
EXE = euler2d

all: $(EXE)

euler2d: euler2d.c
	cc $(CFLAGS) -o $@ $^

clean:
	$(RM) $(EXE)
