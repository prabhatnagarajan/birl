#include <math.h>
#include <stdbool.h>
#include <stdio.h>

static double dot(double* vector1, int index, double* vector2, int size);
double* fast_policy_eval(double V[], double transitions[], int pi[], double rewards[], int num_states, int num_actions, double gamma, double theta, double max_value);

static int fib(int n)
{
	int a = 0;
	int b = 1;
	int i;
	for (i = 0; i < n; i++)
	{
		int temp = b;
		b = b + a;
		a = temp;
	}
	return b;
}

static double logsumexp(double* nums, unsigned int size) {
  double max_exp = nums[0];
  double sum = 0.0;
  unsigned int i;

  for (i = 1 ; i < size ; i++)
  {
    if (nums[i] > max_exp)
      max_exp = nums[i];
   }

  for (i = 0; i < size ; i++)
    sum += exp(nums[i] - max_exp);

  return log(sum) + max_exp;
}

//Assumes the first index stores the length of the array
static double get_index(int* dims, int* indices) {
	int num_dims = dims[0];
	int* partial_prods = malloc(sizeof(int) * num_dims);
	int i;
	for (i = 0; i < num_dims; i++) {
		partial_prods[i] = prod(dims, i + 1, num_dims);
	}
	int total = 0;
	for (i = 0; i < num_dims - 1; i++) {
		total = total + indices[i] * partial_prods[i + 1];
	}
	total = total + indices[num_dims - 1];
	free(partial_prods);
}

int prod(int* dims, int start, int end) {
	int product = 1;
	int i;
	for (i = start; i <= end; i++) {
		product = product * dims[i];
	}
	return product;
}

double* fast_policy_eval(double V[], double transitions[], int pi[], double rewards[], int num_states, int num_actions, double gamma, double theta, double max_value) {
	int index;
	int state = 0;
	double delta = 0;
	double* scalarMult = malloc(sizeof(double) * num_states);
	double* vectorSum = malloc(sizeof(double) * num_states);
	while(true) {
		delta = 0.0;
		for (state = 0; state < num_states; state++) {
			double value = V[state];
			scalar_mult(gamma, V, scalarMult, num_states);
			vector_sum(rewards, scalarMult, vectorSum, num_states);
			index = state * num_states * num_actions + pi[state] * num_states;
			V[state] = dot(transitions, index, vectorSum, num_states);
			delta = fmax(delta, abs(value - V[state]));
			if (V[state] > max_value) {
				free(scalarMult);
				free(vectorSum);
				return V;
			}
			if (V[state] < -max_value){
				free(scalarMult);
				free(vectorSum);
				return V;
			}
		}
		if (delta < theta) {
			break;
		}
	}
	free(scalarMult);
	free(vectorSum);
	return V;
}

static double dot(double* vector1, int index, double* vector2, int size) {
	double total = 0;
	int i;
	for (i = 0; i < size; i++) {
		total = total + vector1[index + i] * vector2[i];
	}
	return total;
}

void scalar_mult(double scalar, double* vector, double* dest, int size) {
	int i;
	for (i = 0; i < size; i++) {
		dest[i] = scalar * vector[i];
	}
	return dest;
}

void vector_sum(double* vector1, double* vector2, double* dest, int size) {
	int i;
	for (i = 0; i < size; i++) {
		dest[i] = vector1[i] + vector2[i];
	}
}

