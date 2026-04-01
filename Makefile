C ?= c++
LD ?= ld
INCLUDE = 

SRC = src/a_string.c src/a_string_slice.c src/util.c src/error.c src/lexer.c src/lexer_types.c
OBJ = $(DEPS) $(SRC:.c=.o)
HEADERS = src/a_vector.h src/common.h $(SRC:.c=.h)

CFLAGS = -Wall -Wextra -pedantic -std=c99 -I./3rdparty
RELEASE_CFLAGS = -O2
DEBUG_CFLAGS = -O0 -ggdb3 -fsanitize=address
TARBALLFILES = Makefile LICENSE.md README.md 3rdparty $(SRC) $(HEADERS) 

TARGET=debug

ifeq (,$(filter clean cleandeps,$(MAKECMDGOALS)))

# goodbye windowze™
ifeq ($(OS),Windows_NT)
$(error building on Windows is not supported.)
endif

ifeq (,$(shell command -v curl))
$(error curl is not installed on your system.)
endif

ifeq (,$(shell command -v unzip))
$(error unzip is not installed on your system.)
endif

ifeq ($(TARGET),debug)
CFLAGS += $(DEBUG_CFLAGS)
else
CFLAGS += $(RELEASE_CFLAGS)
endif

CFLAGS += $(INCLUDE)

endif

beancode: deps $(OBJ) $(HEADERS)
	$(CC) $(CFLAGS) -o beancode src/main.c $(OBJ)

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
