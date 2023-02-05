# **C Neural Network**

This project is an implementation of an Artificial Neural Network (ANN) in C. The ANN is implemented using a struct called **ANN** which contains all the necessary information to define the network, including the number of input, hidden, and output neurons, as well as the weights, biases, and other internal values.

The **ANNNew** function is used to initialize a new ANN with a specified number of input, hidden, and output neurons. The **ANNRandomWeights** function is used to generate random weights for the network, while the **ANNUpdateWeights** function can be used to update the weights and biases of the network with new values.

The **ANNForwardPropagate** function is used to perform the forward propagation step of training the network, which involves calculating the output of the network based on the inputs and the current weights and biases. The **ANNBackwardPropagate** function is used to perform the backward propagation step of training, which involves updating the weights and biases based on the error in the output.

The **ANNBackwardPropagate** function returns a pointer to an array containing the updated weights of the network. This array can be passed to the **ANNUpdateWeights** function to update the weights and biases of the network with the new values.

The **ANNDispose** function can be used to free up memory allocated for the ANN.

This implementation includes following activation functions:

- SIGMOID(x)
- D_SIGMOID(x)
- TANH(x) 
- D_TANH(x)
- ReLU(x) 
- DReLU(x)
- LReLU(x)
- PReLU(a, x)
- ELU(a, x)
- SWISH(x)
- GELU(x) 


You can choose the activation function by passing the name in string format as a parameter to the **ANNForwardPropagate** and **ANNBackwardPropagate** functions.

The **alpha** parameter is used for PReLU and ELU activation functions. It should be a value between 0 and 1.

## **Usage**

1. include "neural\_net.h"
2. Create a new ANN:

```c
ANN\* ann = ANNNew(input\_neurons\_size, hidden\_neurons\_size, hidden\_layer\_size, output\_neurons\_size);
```

3. Generate Random Weights:
```c
ANNRandomWeights(ann, lower, upper);
```
4. Forward Propagation:
```c
ANNForwardPropagate(ann, inputs, outputs, activation\_func, alpha, total\_error);
```
5. Backward Propagation:
```c
ANNBackwardPropagate(ann, inputs, outputs, learning\_rate, activation\_func);
```
6. Update Weights and Biases:
```c
ANNUpdateWeights(ann, weights, biases);
```
1. Dispose of the ANN:
```c
ANNDispose(ann);
```
**Note**: The **inputs** , **outputs** , **weights** , and **biases** arrays should be of the appropriate size for the network. The total\_error array must have enough space to store one double value.

## **Example**

```c
#include "neural_net.h"

int main()
{
	ANN* ann = ANNNew(2, 2, 1, 2);
	double inputs[] = { .05, .10 };
	double outputs[] = { .01, .99 };
	ANNUpdateWeights(ann, (double[]){ .15, .2, .25, .3, .4, .45, .5, .55 }, (double[]){ .35, .60 });

	double* total_error = malloc(sizeof(double));
	ANNForwardPropagate(ann, inputs, outputs, "SIGMOID", 0, total_error);
	ANNBackwardPropagate(ann, inputs, outputs, 0.5, "D_SIGMOID");
	printf("%.9f ", *total_error);
    	ANNDispose(ann);
	scanf_s("%d");
}
```

Once the training is done, the memory allocated for the ANN can be freed using the **ANNDispose** function
