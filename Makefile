CXX ?= c++
LD ?= ld
INCLUDE = 

SRC = src/util.cpp src/error.cpp src/lexer.cpp src/utf8.cpp
DEPS = #3rdparty/simdutf.o
OBJ = $(DEPS) $(SRC:.cpp=.o)
HEADERS = src/common.hpp $(SRC:.cpp=.hpp)

CXXFLAGS = -Wall -Wextra -pedantic -std=c++23 -I./3rdparty
RELEASE_CXXFLAGS = -O2
DEBUG_CXXFLAGS = -O0 -ggdb3 -fsanitize=address
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
CXXFLAGS += $(DEBUG_CXXFLAGS)
else
CXXFLAGS += $(RELEASE_CXXFLAGS)
endif

CXXFLAGS += $(INCLUDE)

endif

beancode: deps $(OBJ) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o beancode src/main.cpp $(OBJ)

%.o: %.c %.h src/common.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

SIMDUTF_VERSION = 8.2.0
dep_simdutf:
	mkdir -p 3rdparty;
	if [ ! -f 3rdparty/simdutf.cpp ]; then\
		cd 3rdparty;\
		curl -fLO https://github.com/simdutf/simdutf/releases/download/v$(SIMDUTF_VERSION)/singleheader.zip;\
		unzip -d singleheader singleheader.zip;\
		cp singleheader/simdutf.h singleheader/simdutf.cpp ./;\
		rm -rf singleheader singleheader.zip;\
	fi

3rdparty/simdutf.o: dep_simdutf
	$(CXX) -c -o 3rdparty/simdutf.o 3rdparty/simdutf.cpp

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
