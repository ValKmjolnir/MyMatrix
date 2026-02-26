.PHONY: all test word2vec

all: test bp.out word2vec.out
test: test.out test_softmax.out

test.out: include/*.hpp test/test.cpp
	c++ -std=c++17 -O3 test/test.cpp -o test.out -I include -fopenmp

test_softmax.out: include/*.hpp test/test_softmax.cpp
	c++ -std=c++17 -O3 test/test_softmax.cpp -o test_softmax.out -I include -fopenmp

bp.out: include/*.hpp src/*.cpp
	c++ -std=c++17 -O3 src/bp.cpp -o bp.out -I include -fopenmp

word2vec.out: include/matrix.hpp include/word2vec.hpp src/word2vec.cpp
	c++ -std=c++17 -O3 src/word2vec.cpp -o word2vec.out -I include -fopenmp

word2vec: word2vec.out