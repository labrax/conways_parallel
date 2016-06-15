#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <pthread.h>

#define SEED

typedef struct _data {
    char * values;
    char * next_values;
    int width;
    int height;
} data;

typedef struct _params {
    long thread_count;
    long rank;
    data * conways_data;
} params;

double rtclock() {
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

int amount_neighbours(data * conways_data, int x, int y) {
    int i, j;
    int amount = 0;
    for(i = y-1; i <= y+1; i++) {
        for(j = x-1; j <= x+1; j++) {
            //printf("%d %d -- %c\n", j, i, conways_data->values[i*conways_data->width+j]);
            if(i == y && j == x)
                continue;
            if(i >= 0 && i < conways_data->height
                    && j >= 0 && j < conways_data->width
                    && conways_data->values[i*conways_data->width+j] == '1') {
                amount++;
            }
        }
    }
    assert(amount >= 0 && amount <= 8);
    return amount;
}

void internal_operate(data * conways_data, int begin_x, int end_x,
        int begin_y, int end_y) {
    int i, j, amount;
    for(i = begin_y; i < end_y; i++) {
        for(j = begin_x; j < end_x; j++) {
            amount = amount_neighbours(conways_data, j, i);
            if(conways_data->values[i*conways_data->width+j] == '1') {
                if(amount < 2 || amount > 3)
                    conways_data->next_values[i*conways_data->width+j] = '0';
                else
                    conways_data->next_values[i*conways_data->width+j] = '1';
            }
            else {
                if(amount == 3)
                    conways_data->next_values[i*conways_data->width+j] = '1';
                else
                    conways_data->next_values[i*conways_data->width+j] = '0';
            }
        }
    }
}

void * internal_pthread(void * thread_params) {
    params * my_params = (params *) thread_params;
    long my_height = my_params->conways_data->height /
        my_params->thread_count;

    internal_operate(my_params->conways_data, 0, my_params->conways_data->width,
            my_params->rank * my_height,
            (my_params->rank + 1) * my_height);

    return NULL;
}

void operate(data * conways_data, int number_threads) {
    pthread_t * pthreads;
    pthreads = (pthread_t *) malloc(sizeof(pthread_t) * number_threads);
    params * thread_params = (params *) malloc(sizeof(params) * number_threads);
    int i;

    for(i = 0; i < number_threads; i++) {
        thread_params[i].thread_count = number_threads;
        thread_params[i].rank = i;
        thread_params[i].conways_data = conways_data;
    }

    for(i = 0; i < number_threads; i++)
        pthread_create(&pthreads[i], NULL, internal_pthread,
                (void *)&thread_params[i]);
    for(i = 0; i < number_threads; i++)
        pthread_join(pthreads[i], NULL);

    free(pthreads);
    free(thread_params);

    char * temp = conways_data->values; //swap buffers
    conways_data->values = conways_data->next_values;
    conways_data->next_values = temp;
    return;
}

void run_n_times(data * conways_data, int iterations, int number_threads) {
    int i;
    for(i = 0; i < iterations; i++)
        operate(conways_data, number_threads);
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
