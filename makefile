.PHONY: all

all: test.out bp.out

test.out: include/*.hpp test/test.cpp
	c++ -std=c++17 -O3 test/test.cpp -o test.out -I include -fopenmp

bp.out: include/*.hpp src/*.cpp
	c++ -std=c++17 -O3 src/bp.cpp -o bp.out -I include -fopenmp