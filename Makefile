C ?= c++
LD ?= ld
INCLUDE = 
DEPS =

SRC = $(wildcard src/*.c) 
OBJ = $(DEPS) $(SRC:.c=.o)
HEADERS = $(wildcard src/*.h)

CFLAGS = -Wall -Wextra -pedantic -std=c99 -I./3rdparty
RELEASE_CFLAGS = -O2
DEBUG_CFLAGS = -O0 -g -fsanitize=address
TARBALLFILES = Makefile LICENSE.md README.md 3rdparty $(SRC) $(HEADERS) 

TARGET=debug

ifeq (,$(filter clean cleandeps,$(MAKECMDGOALS)))

ifeq ($(TARGET),debug)
CFLAGS += $(DEBUG_CFLAGS)
else
CFLAGS += $(RELEASE_CFLAGS)
endif

CFLAGS += $(INCLUDE)

endif

beancode: deps $(OBJ) $(HEADERS)
	$(CC) $(CFLAGS) -o beancode $(OBJ)

%.o: %.c %.h src/common.h
	$(CC) $(CFLAGS) -c -o $@ $<

deps: 

tarball:
	mkdir -p beancode
	cp -r $(TARBALLFILES) beancode/
	tar czf beancode.tar.gz beancode
	rm -rf beancode

distclean: clean cleandeps

clean:
	rm -rf beancode beancode.tar.gz beancode 3rdparty/* $(OBJ)

.PHONY: clean cleanall
