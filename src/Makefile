CC=gcc
CFLAGS=-O3 -fpic -mtune=native -fopenmp -lomp -lgomp -std=c99
LFLAGS=-L.
OBJS=main.o
MAIN=main.x
LIBS=pair_loop_LJ periodic_boundary

all: $(MAIN)

%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

%.o: %.h %.c
	$(CC) -c $(CFLAGS) -o $@ $<

lib%.so: %.o
	$(CC) -shared -o $@ $<

$(MAIN): $(addsuffix .so,$(addprefix lib,$(LIBS)))
	#$(CC) $(LFLAGS) -o $(MAIN) $(OBJS) $(addprefix -l,$(LIBS))

.phony:
clean:
	rm -rf *.o *.so $(MAIN) *.pyc ./build/*

cout:
	rm ./output/*
