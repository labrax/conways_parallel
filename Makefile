CC = gcc
CUDA_CC = nvcc

DEBUG_FLAGS = -Wall -lm -g
RELEASE_FLAGS = -Wall -lm -O3
FLAGS = $(DEBUG_FLAGS)

OPENMP_FLAGS = -fopenmp
PTHREADS_FLAGS = -pthread -lpthread

all: serial openmp openmp_tasks pthreads cuda
	echo 'ready'
	
serial: serial.c
	$(CC) $(FLAGS) serial.c -o serial.out

openmp: openmp.c
	$(CC) $(FLAGS) $(OPENMP_FLAGS) openmp.c -o openmp.out

openmp_taks: openmp_tasks.c
	$(CC) $(FLAGS) $(OPENMP_FLAGS) openmp-tasks.c -o openmp_tasks.out

pthreads:
	$(CC) $(FLAGS) $(PTHREADS_FLAGS) pthreads.c -o pthreads.out

cuda: cuda.cu
	$(CUDA_CC) $(FLAGS) cuda.c -o cuda.out

clean:
	rm *.out
