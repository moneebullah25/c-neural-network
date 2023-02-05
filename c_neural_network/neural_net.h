#ifndef _NEURAL_NET_HEADER_
#define _NEURAL_NET_HEADER_

#include <stdlib.h>
#include <time.h>
#include "activation_functions.c"

#define RANDMAX 0x7ff

// Default arguments aren't allowed in C

typedef struct {
	unsigned int input_neurons_size, hidden_neurons_size, hidden_layer_size, output_neurons_size;
	unsigned int weight_size, bias_size, total_neurons;
	double *inputs, *weights, *in_outputs, *out_outputs, *biases, *deltas;
	double** in_hiddens; double** out_hiddens; // If more than one hidden layer otherwise treated as double* hiddens
} ANN;

// Initialize Artificial Neural Network. Update the cirre
ANN* ANNNew(unsigned int input_neurons_size, unsigned int hidden_neurons_size,
	unsigned int hidden_layer_size, unsigned int output_neurons_size);

// Generate Random Weights from lower to upper inclusive
void ANNRandomWeights(ANN* ann, double lower, double upper);

// Update weights based on given array and then free the passed array
void ANNUpdateWeights(ANN* ann, double* weights, double* biases);

/*Copy contents of Total Error E value to total_error after Forward Propogating once
E = SUMMATION(1/2 * (target - output)^2) :: Delta = (target - output)
For PReLU and ELU Alpha must be provided [0, 1] 
For rest of Activation Functions Alpha can be of any value since not used*/
void ANNForwardPropagate(ANN* ann, double const *inputs, double const *outputs, 
	char* activation_func, double alpha, double* total_error);

// Return weights list after Back Propagating once and also update the existing weights
double* ANNBackwardPropagate(ANN* ann, double const* inputs, double const* outputs, double learning_rate, char* activation_func);

// Disposes of all allocated memory for ANN
void ANNDispose(ANN* ann);

#endif