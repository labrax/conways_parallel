#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>

#define SEED

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

int amount_neighbours(data * conways_data, int x, int y) {
	int i, j;
	int amount = 0;

	#pragma omp parallel
	{
		#pragma omp single
		{
			for(i = y-1; i <= y+1; i++) {
				for(j = x-1; j <= x+1; j++) {
					//printf("%d %d -- %c\n", j, i, conways_data->values[i*conways_data->width+j]);
					#pragma omp task firstprivate(i, j, x, y)
					{
						if((i != y || j != x) && i >= 0 && i < conways_data->height
								&& j >= 0 && j < conways_data->width
								&& conways_data->values[i*conways_data->width+j] == '1') {
							#pragma omp critical
							amount++;
						}
					}
				}
			}
		}
	}

	assert(amount >= 0 && amount <= 8);
	return amount;
}

void operate(data * conways_data) {
	int i, j, amount;
	for(i = 0; i < conways_data->height; i++) {
		for(j = 0; j < conways_data->width; j++) {
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

	char * temp = conways_data->values; //swap buffers
	conways_data->values = conways_data->next_values;
	conways_data->next_values = temp;
	return;
}

void run_n_times(data * conways_data, int iterations) {
    int i;
	for(i = 0; i < iterations; i++)
		operate(conways_data);
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
	int w, h, seed;
	data conways_data;
	scanf(" %d %d", &w, &h);
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
	run_n_times(&conways_data, iterations);
	end = rtclock();

	print_data(&conways_data);
	printf("%f\n", end-start);

	free(conways_data.values);
	free(conways_data.next_values);
	return 0;
}
