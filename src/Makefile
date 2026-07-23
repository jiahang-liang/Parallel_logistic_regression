CXX = g++
MPICXX = mpic++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra
OMPFLAGS = -fopenmp

all: seq omp mpi

seq: main_seq.cpp common.cpp
	$(CXX) $(CXXFLAGS) main_seq.cpp common.cpp -o seq

omp: main_omp.cpp common.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) main_omp.cpp common.cpp -o omp

mpi: main_mpi.cpp common.cpp
	$(MPICXX) $(CXXFLAGS) main_mpi.cpp common.cpp -o mpi

clean:
	rm -f seq omp mpi
