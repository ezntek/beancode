CXX ?= c++
LD ?= ld
INCLUDE = 

SRC = src/main.cpp src/lexer.cpp 
OBJ = $(SRC:.c=.o)
HEADERS = src/lexer.hpp

CXXFLAGS = -Wall -Wextra -pedantic
RELEASE_CXXFLAGS = -O2
DEBUG_CXXFLAGS = -D_A_STRING_DEBUG -O0 -ggdb3 -fsanitize=address
TARBALLFILES = Makefile LICENSE.md README.md 3rdparty $(SRC) $(HEADERS) main.c 

TARGET=debug

ifeq (,$(filter clean cleandeps,$(MAKECMDGOALS)))

# goodbye windowze™
ifeq ($(OS),Windows_NT)
$(error building on Windows is not supported.)
endif

ifeq (,$(shell command -v curl))
$(error curl is not installed on your system.)
endif

ifeq (,$(shell command -v qbe))
$(error qbe is not installed on your system.)
endif

ifeq ($(TARGET),debug)
CXXFLAGS += $(DEBUG_CXXFLAGS)
else
CXXFLAGS += $(RELEASE_CXXFLAGS)
endif

CXXFLAGS += $(INCLUDE)

endif

beancode: deps $(OBJ) $(HEADERS) main.o
	$(CXX) $(CXXFLAGS) -o beancode main.o $(OBJ)

main.o: main.c common.h
	$(CXX) -c $(CXXFLAGS) -o $@ $<

%.o: %.c %.h common.h
	$(CXX) -c $(CXXFLAGS) -o $@ $<

dep_uthash:
	mkdir -p 3rdparty/
	if [ ! -f 3rdparty/uthash.h ]; then \
		curl -fL -o 3rdparty/uthash.h https://raw.githubusercontent.com/troydhanson/uthash/refs/heads/master/src/uthash.h; \
	fi

deps: dep_uthash

tarball:
	mkdir -p beancode
	cp -r $(TARBALLFILES) beancode/
	tar czf beancode.tar.gz beancode
	rm -rf beancode

distclean: clean cleandeps

clean:
	rm -rf beancode beancode.tar.gz beancode $(OBJ) main.o

.PHONY: clean cleanall
