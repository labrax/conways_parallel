CC = gcc
CUDA_CC = /usr/local/cuda-7.5/bin/nvcc

DEBUG_FLAGS = -Wall -Wextra -lm -g
RELEASE_FLAGS = -Wall -Wextra -lm -O3
FLAGS = $(RELEASE_FLAGS)

OPENMP_FLAGS = -fopenmp
PTHREADS_FLAGS = -pthread -lpthread

.PHONY: all run clean

all: serial openmp openmp_tasks pthreads cuda cuda_shared
	@echo 'ready'
	
serial: serial.c
	$(CC) $(FLAGS) serial.c -o serial

openmp: openmp.c
	$(CC) $(FLAGS) $(OPENMP_FLAGS) openmp.c -o openmp

openmp_tasks: openmp_tasks.c
	$(CC) $(FLAGS) $(OPENMP_FLAGS) openmp_tasks.c -o openmp_tasks

pthreads:
	$(CC) $(FLAGS) $(PTHREADS_FLAGS) pthreads.c -o pthreads

cuda: cuda.cu
ifneq ("$(wildcard $(CUDA_CC))", "")
	$(CUDA_CC) cuda.cu -o cuda
endif

cuda_shared: cuda_shared.cu
ifneq ("$(wildcard $(CUDA_CC))", "")
	$(CUDA_CC) cuda_shared.cu -o cuda_shared
endif

run: all
	@(cd input; ./test.py -r)

compare: all
	@(cd input; ./test.py -r -c)

clean:
	rm -f serial openmp openmp_tasks pthreads cuda cuda_shared
