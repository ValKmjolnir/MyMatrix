.PHONY: all test

all: test bp.out
test: test.out test_softmax.out

test.out: include/*.hpp test/test.cpp
	c++ -std=c++17 -O3 test/test.cpp -o test.out -I include -fopenmp

test_softmax.out: include/*.hpp test/test_softmax.cpp
	c++ -std=c++17 -O3 test/test_softmax.cpp -o test_softmax.out -I include -fopenmp

bp.out: include/*.hpp src/*.cpp
	c++ -std=c++17 -O3 src/bp.cpp -o bp.out -I include -fopenmp