# **Usuage**
To compile the project and create the executables, run the following command:
```
make
```
This will compile all the source files, create object files, target library, and executable file for tests. This will generate following output:
```
└─$ make 
```

## **Running tests**
To run the tests, run the following command:
```
make run_tests
```
This will compile and run all the tests in the tests directory. You should see the following output:
```
└─$ make run_tests
ar rcs lib/libc-ann-matrix.a obj/af.o obj/ann.o obj/c_matrix.o
gcc -Wall -Werror -I includes -o tests/bin/t_ann_1.out tests/t_ann_1.c -L ./lib -lc-ann-matrix -lm
for test in tests/bin/t_ann_1.out; do \
        ./$test; \
done
```

# **Introduction**

This code is an implementation of a artificial neural network (ANN) using a single or multiple hidden layer with a configurable number of neurons. The activation functions for the hidden and output layers are configurable, as well as their derivatives.

The ANN structure is defined by the ANN struct, which holds the sizes of the input, hidden, and output layers, as well as the activation functions and the weights and biases. The weights and biases are initialized randomly using ANNRandomWeights() and can be updated using ANNUpdateWeights(). The forward propagation is implemented using ANNForwardPropagate(), which takes the inputs and propagates them through the network to produce the outputs.

The error function is not implemented in this code. It may be implemented separately as a loss function and minimized using techniques like backpropagation.

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

## **Functions**

- **ANNRandomWeights(ANN \*ann, double lower, double upper)**: This function generates a matrix of random weights to initialize the neural network. It takes two arguments: the lower bound and upper bound and generates the random weights within that range.
- **ANNUpdateWeights(ANN \*ann, double \*weights, double \*biases)**: This function updates the weights given the array of weights and biases passed as argument to the function. Then it will modify those changes to the ANN struct. 
- **ANNForwardPropagate(ANN \*ann, double \*inputs)**: - This function takes an ANN structure pointer and a pointer to an array of double inputs as its arguments. The ANN structure represents the neural network, and the input array contains the inputs to the network. The function performs a forward propagation through the network, computing the output of each neuron and passing it as an input to the next layer until it reaches the output layer. The function does not return any value, but it updates the output array of the ANN structure with the output of the network.
- **ANNTotalError(ANN \*ann, double const \*outputs, double \*result)**: This function takes an ANN structure pointer, a pointer to an array of double outputs from the neural network, and a pointer to a double variable to store the result as its arguments. The function computes the total error between the expected outputs and the actual outputs of the neural network. It does so by comparing each output of the network to the corresponding expected output and summing the squared difference of each output. The resulting total error is stored in the result variable
- **ANNBackwardPropagate(ANN \*ann, double const \*inputs, double const \*outputs, double learning_rate)**: This function takes an ANN structure pointer, a pointer to an array of double inputs to the neural network, a pointer to an array of double expected outputs, and a learning rate as its arguments. The function performs a backward propagation through the network, adjusting the weights and biases of each neuron to minimize the error. The function returns a pointer to a list of weights after backpropagating once and updates the existing weights of the network. The learning rate controls how much the weights are adjusted during each iteration of backpropagation.
