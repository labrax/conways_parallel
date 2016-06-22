#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

#define SEED
#define BLOCK_SIZE 16

typedef struct _data {
    char * values;
    char * next_values;
    int width;
    int height;
} data;

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

__global__ void  operate(char * source, char * goal, int width, int height) {
    __shared__ char local[pow(BLOCK_SIZE+2, 2)];
    int index_i = blockDim.x * blockIdx.x + threadIdx.x;
    int index_j = blockDim.y * blockIdx.y + threadIdx.y;
    int i, j;

    int index = index_i*width + index_j;
    int local_index = BLOCK_SIZE + 2*(threadIdx.y+1) + 1 +
        threadIdx.y*BLOCK_SIZE + threadIdx.x;

    /* Mapeamento um-pra-um dos elementos */
    local[local_index] = source[index];

    /* Mapeamento do canto superior esquerdo */
    if(threadIdx.x == 0 && threadIdx.y == 0) {

        /* Elemento da diagonal superior esquerda */
        if((index - width - 1) >= 0 && (index - width - 1) < width*height) {
            local[local_index - (BLOCK_SIZE+2) - 1] = source[index - width - 1];
        }

        /* Elemento acima */
        if((index - width) >= 0 && (index - width) < width*height) {
            local[local_index - (BLOCK_SIZE+2)] = source[index - width];
        }

        /* Elemento do lado esquerdo */
        if((index - 1) >= 0 && (index - 1) < width*height) {
            local[local_index - 1] = source[index - 1];
        }
    }

    /* Mapeamento do canto superior direito */
    else if(threadIdx.x == BLOCK_SIZE-1 && threadIdx.y == 0) {

        /* Elemento da diagonal superior direita */
        if((index - width + 1) >= 0 && (index - width + 1) < width*height) {
            local[local_index - (BLOCK_SIZE+2) + 1] = source[index - width + 1];
        }

        /* Elemento acima */
        if((index - width) >= 0 && (index - width) < width*height) {
            local[local_index - (BLOCK_SIZE+2)] = source[index - width];
        }

        /* Elemento do lado direito */
        if((index + 1) >= 0 && (index + 1) < width*height) {
            local[local_index + 1] = source[index + 1];
        }
    }

    /* Mapeamento do canto inferior esquerdo */
    else if(threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE-1) {

        /* Elemento da diagonal inferior esquerda */
        if((index + width - 1) >= 0 && (index + width - 1) < width*height) {
            local[local_index + (BLOCK_SIZE+2) - 1] = source[index + width - 1];
        }

        /* Elemento abaixo */
        if((index + width) >= 0 && (index + width) < width*height) {
            local[local_index + (BLOCK_SIZE+2)] = source[index + width];
        }

        /* Elemento do lado esquerdo */
        if((index - 1) >= 0 && (index - 1) < width*height) {
            local[local_index - 1] = source[index - 1];
        }
    }

    /* Mapeamento do canto inferior direito */
    else if(threadIdx.x == BLOCK_SIZE-1 && threadIdx.y == BLOCK_SIZE-1) {

        /* Elemento da diagonal inferior direita */
        if((index + width + 1) >= 0 && (index + width + 1) < width*height) {
            local[local_index + (BLOCK_SIZE+2) + 1] = source[index + width + 1];
        }

        /* Elemento abaixo */
        if((index + width) >= 0 && (index + width) < width*height) {
            local[local_index + (BLOCK_SIZE+2)] = source[index + width];
        }

        /* Elemento do lado direito */
        if((index + 1) >= 0 && (index + 1) < width*height) {
            local[local_index + 1] = source[index + 1];
        }
    }

    __syncthreads();

    if (index_i < height && index_j < width && index < height*width) {
        int amount = amount_neighbours(local, threadIdx.y, threadIdx.x,
                BLOCK_SIZE, BLOCK_SIZE);
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
    scanf(" %d %d %d", &w, &h, &number_threads);
    conways_data.width = w;
    conways_data.height = h;
    conways_data.values = (char *) malloc(sizeof(char) * w * h);
    conways_data.next_values = (char *) malloc(sizeof(char) * w * h);

    #ifdef SEED
    scanf(" %d", &seed);
    srand(seed);
    #endif
    
    int i, j;
    for(i = 0; i < h; i++) {
        for(j = 0; j < w; j++) {
            #ifdef SEED
            conways_data.values[i * w + j] = '0' + rand() % 2;
            #else
            scanf(" %c", &conways_data.values[i * w + j]);
            #endif
        }
    }

    int iterations;
    scanf(" %d", &iterations);

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
