CC=gcc
CFLAGS=-O3 -fpic -mtune=native -fopenmp -lomp -lgomp -std=c99
LFLAGS=-L.
OBJS=main.o
MAIN=main.x
LIBS=

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
ifeq ($(strip $(BUILD_DIR)),)
	rm -rf *.o *.so $(MAIN) *.pyc ./build/*
else
	rm -rf *.o *.so *.pyc $(BUILD_DIR)*
endif

cout:
	rm ./output/*

chart:
	pyreverse -a1 -s1 -f ALL -my -o png *.py
