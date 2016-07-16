#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

#define SEED
#define BLOCK_SIZE 32

typedef struct _data {
    char * values;
    char * next_values;
    int width;
    int height;
} data;

void input_error() {
    fprintf(stderr, "Erro na leitura dos parâmetros");
    exit(EXIT_FAILURE);
}

void mem_error() {
    fprintf(stderr, "Erro na alocação de memória");
    exit(EXIT_FAILURE);
}

double rtclock() {
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

__device__ int amount_neighbours(char * source, int x, int y, int width, int height) {
    int i, j;
    int amount = 0;
    for(i = y-1; i <= y+1; i++) {
        for(j = x-1; j <= x+1; j++) {
            //printf("%d %d -- %c\n", j, i, conways_data->values[i*conways_data->width+j]);
            if(i == y && j == x)
                continue;
            if(i >= 0 && i < height
                    && j >= 0 && j < width
                    && source[i*width+j] == '1') {
                amount++;
            }
        }
    }
    assert(amount >= 0 && amount <= 8);
    return amount;
}

#define MASK_RADIUS 1
#define MASK_WIDTH 3
__global__ void operate(char * source, char * goal, int sizex, int sizey) {
	__shared__ char local[BLOCK_SIZE + MASK_WIDTH - 1][BLOCK_SIZE + MASK_WIDTH - 1];
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = i * sizex + j;

	int prim_x = j - MASK_RADIUS;
	int first_x = prim_x;
	for(; first_x - prim_x + threadIdx.x < MASK_WIDTH + BLOCK_SIZE - 1; first_x += BLOCK_SIZE) {	
		int prim_y = i - MASK_RADIUS;
		int first_y = prim_y;
		for(; first_y - prim_y + threadIdx.y < MASK_WIDTH + BLOCK_SIZE - 1; first_y += BLOCK_SIZE) {
			if(first_y >= 0 && first_y < sizey && first_x >= 0 && first_x < sizex) {
				local[first_y - prim_y + threadIdx.y][first_x - prim_x + threadIdx.x] =
					 source[first_y * sizex + first_x];
			}
			else {
				local[first_y - prim_y + threadIdx.y][first_x - prim_x + threadIdx.x] = '0';
			}
		}
	}
    __syncthreads();

	if(i < sizey && j < sizex) {
		int l_j, l_i;
		int amount = 0;

		for(l_i = 0; l_i < MASK_WIDTH; l_i++) {
			if( ( (int) threadIdx.y + l_i >= 0 ) && ( (int) threadIdx.y + l_i < BLOCK_SIZE + MASK_WIDTH - 1) ) {
				for(l_j = 0; l_j < MASK_WIDTH; l_j++){
					if( ( (int) threadIdx.x + l_j >= 0 ) && ( (int) threadIdx.x + l_j < BLOCK_SIZE + MASK_WIDTH - 1) ) {
                        if(local[threadIdx.y + l_i][threadIdx.x + l_j] == '1')
                            amount++;
					}
				}
			}
		}

        if(source[index] == '1')
            amount--;

        if(source[index] == '1') {
            if(amount < 2 || amount > 3)
                goal[index] = '0';
            else
                goal[index] = '1';
        }
        else {
            if(amount == 3)
                goal[index] = '1';
            else
                goal[index] = '0';
        }
	}
}

void run_n_times(data * conways_data, int iterations, int number_threads) {
    int i;
    
    char * d_A, * d_B;

    int size = conways_data->height * conways_data->width * sizeof(char);
    cudaMalloc((void**) &d_A, size);
    cudaMalloc((void**) &d_B, size);
    
    cudaMemcpy(d_A, conways_data->values, size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(ceil(conways_data->width/(float) threadsPerBlock.x), ceil(conways_data->height/(float) threadsPerBlock.y));
    
    for(i = 0; i < iterations; i++) {
        operate<<<numBlocks, threadsPerBlock>>>(i%2 == 0? d_A : d_B, i%2 == 0? d_B : d_A, conways_data->width, conways_data->height);
        cudaThreadSynchronize();
    }
    
    cudaMemcpy(conways_data->values, i%2 == 0? d_A : d_B, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    return;
}

void print_data(data * conways_data) {
    int i, j;
    for(i = 0; i < conways_data->height; i++) {
        for(j = 0; j < conways_data->width; j++) {
            printf("%c ", conways_data->values[i*conways_data->width+j]);
        }
        printf("\n");
    }
    return;
}

int main(void) {
    int w, h, number_threads, seed;
    data conways_data;
    if(scanf(" %d %d %d", &w, &h, &number_threads) != 3) {
        input_error();
    }
    conways_data.width = w;
    conways_data.height = h;
    conways_data.values = (char *) malloc(sizeof(char) * w * h);
    conways_data.next_values = (char *) malloc(sizeof(char) * w * h);

    if(conways_data.values == NULL || conways_data.next_values == NULL) {
        mem_error();
    }

    #ifdef SEED
    if(scanf(" %d", &seed) != 1) {
        input_error();
    }
    srand(seed);
    #endif
    
    int i, j;
    for(i = 0; i < h; i++) {
        for(j = 0; j < w; j++) {
            #ifdef SEED
            conways_data.values[i * w + j] = '0' + rand() % 2;
            #else
            if(scanf(" %c", &conways_data.values[i * w + j]) != 1) {
                input_error();
            }
            #endif
        }
    }

    int iterations;
    if(scanf(" %d", &iterations) != 1) {
        input_error();
    }

    double start, end;
    start = rtclock();
    run_n_times(&conways_data, iterations, number_threads);
    end = rtclock();

    print_data(&conways_data);
    printf("%f\n", end-start);

    free(conways_data.values);
    free(conways_data.next_values);
    return 0;
}
