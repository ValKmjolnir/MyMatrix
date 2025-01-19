test.out: library/*.hpp src/*.cpp
	c++ -std=c++17 -O3 src/test.cpp -o test.out -I library -fopenmp