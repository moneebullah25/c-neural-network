#ifndef _NEURAL_NET_HEADER_
#define _NEURAL_NET_HEADER_

#define RANDMAX 0x7ff

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
void ANNBackwardPropagate(ANN* ann, double const* inputs, double const* outputs, double learning_rate, char* activation_func);

// The function takes in a filename as a string and two integer pointers to return 
// the number of rows and columns in the CSV data. It returns a two-dimensional array 
// of doubles representing the data in the CSV file. You can then use this data to train your ANN.
double** ANNReadCSV(char* filename, unsigned int* nrows, unsigned int* ncols);


void ANNTrain(ANN* ann, double const* inputs, double const* outputs, double alpha, double learning_rate,
	char* activation_func, char* d_activation_func, double* total_error);

// Function to split the dataset into train and test sets
void ANNTrainTestSplit(double** dataset, unsigned int num_rows, unsigned int num_cols, double ratio,
	double** train_return, double** test_return);

// Function to extract the label column from the dataset
void ANNLabelExtract(double** dataset, unsigned int label_col_index, unsigned int num_rows, unsigned int num_cols,
	double* data_ret);

void ANNDeleteFeature(double** dataset, unsigned int feature_index, unsigned int num_rows, unsigned int num_cols);

// Disposes of all allocated memory for ANN
void ANNDispose(ANN* ann);

#endif