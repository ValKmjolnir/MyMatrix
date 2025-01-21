.PHONY: all

all: test.out bp.out

test.out: include/*.hpp src/*.cpp
	c++ -std=c++17 -O3 src/test.cpp -o test.out -I include -fopenmp

bp.out: include/*.hpp src/*.cpp
	c++ -std=c++17 -O3 src/bp.cpp -o bp.out -I include -fopenmp