#include "neural_net.h"

ANN* ANNNew(unsigned int input_neurons_size, unsigned int hidden_neurons_size, 
	unsigned int hidden_layer_size, unsigned int output_neurons_size)
{
	ANN* ann = malloc(sizeof(ANN));
	ASSERT(ann && input_neurons_size && hidden_neurons_size && hidden_layer_size && output_neurons_size);
	ann->input_neurons_size = input_neurons_size;
	ann->hidden_neurons_size = hidden_neurons_size;
	ann->hidden_layer_size = hidden_layer_size;
	ann->output_neurons_size = output_neurons_size;
	ann->weight_size = (input_neurons_size*hidden_neurons_size) + (hidden_neurons_size*
		(hidden_layer_size - 1)) + (output_neurons_size*hidden_neurons_size);
	ann->bias_size = ann->hidden_layer_size + 1;

	ann->inputs = malloc(ann->input_neurons_size*sizeof(double));
	ann->in_hiddens = malloc(ann->hidden_layer_size*sizeof(double*));
	for (unsigned int i = 0; i < ann->hidden_layer_size; i++)
		(ann->in_hiddens)[i] = malloc(ann->hidden_neurons_size*sizeof(double));
	ann->out_hiddens = malloc(ann->hidden_layer_size*sizeof(double*));
	for (unsigned int i = 0; i < ann->hidden_layer_size; i++)
		(ann->out_hiddens)[i] = malloc(ann->hidden_neurons_size*sizeof(double));
	ann->in_outputs = malloc(ann->output_neurons_size*sizeof(double));
	ann->out_outputs = malloc(ann->output_neurons_size*sizeof(double));
	ann->weights = malloc(ann->weight_size*sizeof(double));
	ann->biases = malloc(ann->bias_size*sizeof(double));
	ann->deltas = malloc(ann->output_neurons_size*sizeof(double));
	ann->total_neurons = input_neurons_size + (hidden_neurons_size*hidden_layer_size) 
		+ output_neurons_size;

	return ann;
}

void ANNRandomWeights(ANN* ann, double lower, double upper)
{
	ASSERT(ann && (upper > lower));
	srand(time(NULL));   // Initialization, should only be called once.
	for (unsigned int i = 0; i < ann->weight_size; i++)
	{
		double randD = lower + (rand() / (double)RAND_MAX) * (upper - lower);
		ann->weights[i] = randD;
	}
	for (unsigned int i = 0; i < ann->bias_size; i++)
	{
		double randD = lower + (rand() / (double)RAND_MAX) * (upper - lower);
		ann->biases[i] = randD;
	}
}

void ANNUpdateWeights(ANN* ann, double* weights, double* biases)
{
	ASSERT(ann && weights && biases);
	for (unsigned int i = 0; i < ann->weight_size; i++)
		ann->weights[i] = weights[i];
	for (unsigned int i = 0; i < ann->bias_size; i++)
		ann->biases[i] = biases[i];
}

double ActivationFunction(double input, double alpha, char* activation_func)
{
	ASSERT(activation_func);
	if (activation_func == "SIGMOID")
		return SIGMOID(input);
	else if (activation_func == "D_SIGMOID")
		return D_SIGMOID(input);
	else if (activation_func == "TANH")
		return TANH(input);
	else if (activation_func == "D_TANH")
		return D_TANH(input);
	else if (activation_func == "ReLU")
		return ReLU(input);
	else if (activation_func == "DReLU")
		return DReLU(input);
	else if (activation_func == "LReLU")
		return LReLU(input);
	else if (activation_func == "PReLU")
		return PReLU(input, alpha);
	else if (activation_func == "ELU")
		return ELU(input, alpha);
	else if (activation_func == "SWISH")
		return SWISH(input);
	else if (activation_func == "GELU")
		return GELU(input);
	else return -DOUBLE_MAX;
}

void* memory_copy(void* dest, const void* src, unsigned int n)
{
	ASSERT(n > 0);
	char *csrc = (char *)src;
	char *cdest = (char *)dest;

	for (unsigned int i = 0; i<n; i++)
		cdest[i] = csrc[i];
	return dest;
}

void ANNForwardPropagate(ANN* ann, double const *inputs, double const *outputs,
	char* activation_func, double alpha, double* total_error)
{
	ASSERT(ann && inputs && outputs && activation_func);
	if (ActivationFunction(1.0, 0.5, activation_func) == -DOUBLE_MAX)
	{
		printf("ACTIVATION FUNCTION NOT DEFINED\nPLEASE PROVIDE DEFINATION IN \"activation_functions.c\"\n");
		return;
	}

	// Assigning Inputs 
	for (unsigned int i = 0; i < ann->input_neurons_size; i++)
		ann->inputs[i] = inputs[i];
	// Hidden Layer Calculations
	unsigned int weight_index = 0;
	for (unsigned int h = 0; h < ann->hidden_layer_size; h++){
		for (unsigned int i = 0; i < ann->hidden_neurons_size; i++)
		{
			if (h == 0) // Input to First Hidden Layers
			{ 
				ann->in_hiddens[h][i] = 0;
				for (unsigned int j = 0; j < ann->input_neurons_size; j++)
				{
					ann->in_hiddens[h][i] += ann->inputs[j] * ann->weights[weight_index++];
				}
				ann->in_hiddens[h][i] += ann->biases[h];
				ann->out_hiddens[h][i] = ActivationFunction(ann->in_hiddens[h][i], alpha, activation_func);
			}
			else // First Hidden Layer to Preceeding Hidden Layer
			{
				ann->in_hiddens[h][i] = 0;
				for (unsigned int j = 0; j < ann->hidden_neurons_size; j++)
				{
					ann->in_hiddens[h][i] += ann->in_hiddens[h - 1][i] * ann->weights[weight_index++];
				}
				ann->in_hiddens[h][i] += ann->biases[h];
				ann->out_hiddens[h][i] = ActivationFunction(ann->in_hiddens[h][i], alpha, activation_func);
			}
		}
	}
	// Output Layer Calculations
	for (unsigned int i = 0; i < ann->output_neurons_size; i++)
	{
		ann->in_outputs[i] = 0;
		for (unsigned int j = 0; j < ann->hidden_neurons_size; j++)
		{
			ann->in_outputs[i] += ann->out_hiddens[ann->hidden_layer_size - 1][j] * ann->weights[weight_index++];
		}
		ann->in_outputs[i] += ann->biases[ann->hidden_layer_size];
		ann->out_outputs[i] = ActivationFunction(ann->in_outputs[i], alpha, activation_func);
	}
	double error = 0;
	// Calculating Deltas
	for (unsigned int i = 0; i < ann->output_neurons_size; i++)
	{
		ann->deltas[i] = outputs[i] - ann->out_outputs[i];
		error += 0.5*(pow(ann->deltas[i], 2));
	}
	// Returning total_error
	memory_copy(total_error, &error, sizeof(double));
	
}

double* ANNBackwardPropagate(ANN* ann, double const *inputs, double const *outputs, double learning_rate, char* activation_func)
{
	/* Backpropogate from output neuron to last hidden layer finding pd(E)/pd(Wi)
	:: pd : Partial Derivative, E : Total Error, Wi: Weight at index i */
	unsigned int weight_index = ann->weight_size - 1;
	unsigned int rel_weight_index = 0;
	double* new_weights = (double *)malloc(ann->weight_size*sizeof(double));
	ASSERT(new_weights);
	
	// Weight Updation from Output layer to last hidden layer
	for (long int i = ann->output_neurons_size - 1; i >= 0; i--)
	{
		double dEt_doi = -(outputs[i] - ann->out_outputs[i]); // delta(E_total)/delta(out_output_i)
		double doi_dni = ActivationFunction(ann->in_outputs[i], 0, activation_func); // delta(out_output_i)/delta(net_output_i)
		for (long int j = ann->hidden_neurons_size - 1; j >= 0; j--)
		{
			new_weights[weight_index] = learning_rate*(dEt_doi * doi_dni * ann->out_hiddens[ann->hidden_layer_size - 1][j]);
			new_weights[weight_index] = ann->weights[weight_index] - new_weights[weight_index];
			weight_index--;
		}
	}

	/* Backpropogate Hidden Layer Calculations finding pd(E)/pd(Wi)
	:: pd : Partial Derivative, E : Total Error, Wi: Weight at index i */
	for (long int g = ann->input_neurons_size - 1; g >= 0; g--) {
		for (long int h = ann->hidden_layer_size - 1; h >= 0; h--) {
			for (long int i = ann->hidden_neurons_size - 1; i >= 0; i--)
			{
				if (h == 0 && ann->hidden_layer_size == 1) // Input to First Hidden Layers
				{
					double dEt_doi = 0; // delta(E_total)/delta(out_output_i)
					for (long int j = ann->output_neurons_size - 1; j >= 0; j--)
					{
						dEt_doi += -(outputs[j] - ann->out_outputs[j]) *
							ActivationFunction(ann->in_outputs[j], 0, activation_func) *
							ann->weights[(ann->input_neurons_size*ann->hidden_neurons_size) + (i + (j * 2))];
					}
					dEt_doi *= ActivationFunction(ann->in_hiddens[h][i], 0, activation_func) * ann->inputs[g];
					new_weights[weight_index] = learning_rate*dEt_doi;
					new_weights[weight_index] = ann->weights[weight_index] - new_weights[weight_index];
					weight_index--;
				}
				else // First Hidden Layer to Preceeding Hidden Layer
				{
					
				}
			}
		}
	}

	for (unsigned int i = 0; i < ann->weight_size; i++)
		ann->weights[i] = new_weights[i];

	return new_weights;
}

double* ANNBackwardPropagate(ANN* ann, double const *inputs, double const *outputs, double learning_rate, char* activation_func)
{
	// Updating Hidden Neurons Weights
	for (long int on = ann->output_neurons_size - 1; on >= 0; on--)
	{
		for (long int hl = ann->hidden_layer_size - 1; hl >= 0; hl--)
		{
			for (long int hn = ann->hidden_neurons_size - 1; hn >= 0; hn--)
			{
				for (long int in = ann->input_neurons_size - 1; in >= 0; in--)
				{

				}
			}
		}
	}
	

	// Updating Input to First Hidden Neurons Weights

}