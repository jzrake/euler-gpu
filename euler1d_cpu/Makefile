CFLAGS = -Wall -O3 -lm
EXE = euler1d

all: $(EXE)

euler1d: euler1d.c
	cc $(CFLAGS) -o $@ $^

clean:
	$(RM) $(EXE)
