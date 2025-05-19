CC = cc
INCLUDE = -I./3rdparty/asv/
LIBS =

OBJ = 3rdparty/asv/asv.o
OBJ += main.o

RELEASE_CFLAGS = -O2 -Wall -Wextra -pedantic -march=native -flto=auto $(INCLUDE) $(LIBS)
DEBUG_CFLAGS = -O0 -g -Wall -Wextra -pedantic -fsanitize=address $(INCLUDE) $(LIBS)
TARBALLFILES = Makefile LICENSE.md README.md *.h *.c 3rdparty assets

HEADERS = *.h 

TARGET=debug

ifeq ($(TARGET),debug)
	CFLAGS=$(DEBUG_CFLAGS)
else
	CFLAGS=$(RELEASE_CFLAGS)
endif

beancode: setup $(OBJ)
	$(CC) $(CFLAGS) -o beancode $(OBJ)

setup: deps settings

settings:

deps: 
	test -d 3rdparty/asv || git submodule --init --recursive
	test -f 3rdparty/asv/asv.o || make -C 3rdparty/asv
	
updatedeps:
	rm -rf 3rdparty/*
	make deps

tarball: deps
	mkdir -p beancode
	cp -rv $(TARBALLFILES) beancode/
	tar czvf beancode.tar.gz beancode
	rm -rf beancode

defaults:
	rm -f settings.h
	cp settings.def.h settings.h

clean:
	rm -rf beancode beancode.tar.gz beancode $(OBJ)
	rm -f 3rdparty/include/*

cleanall: clean defaults

.PHONY: clean cleanall
