#include <math.h>
#include <stdio.h>


#ifndef NULL 
#define NULL 0
#endif

#ifdef DEBUG
#define ASSERT(x) ((void)(!(x) && assert_handler(#x, __FILE__, __LINE__) && (HALT(), 1)))
#else
#define ASSERT(x) ((void)sizeof(x))
#endif

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MULT(x, y) (x * y)

#define EULER_CONST 2.71828182845904523536
#define PIE_CONST 3.14159265358979323846
#define DOUBLE_MAX 1.7976931348623158e+308 

#define SIGMOID(x) 1/(1 + pow(EULER_CONST, -x))
#define D_SIGMOID(x) ((SIGMOID(x))*(1-SIGMOID(x)))
#define TANH(x) ((pow(EULER_CONST, x) - pow(EULER_CONST, -x))/(pow(EULER_CONST, x) + pow(EULER_CONST, -x)))
#define D_TANH(x) (1-pow(TANH(x), 2))
#define ReLU(x) (MAX(0, x))
#define DReLU(x) (x >= 0 ? 1 : 0) // Dying ReLU
#define LReLU(x) (MAX(MULT(0.1, x), x)) // Leaky ReLU
#define PReLU(a, x) (MAX(MULT(a, x), x)) //Parametric ReLU
#define ELU(a, x) (x >= 0 ? x : MULT(a, pow(EULER_CONST, x) - 1)) // Exponential Linear Units
#define SWISH(x) (x * SIGMOID(x)) // Self-gated activation function
#define GELU(x) (MULT(MULT(0.5, x), 1 + TANH(MULT(sqrt(2/PIE_CONST), x + MULT(0.044715, pow(x, 3)))))) // Gaussian Error Linear Unit (GELU)

static void SOFTMAX(double *input, unsigned int input_len) {
	ASSERT(input);
	double m = (double)(-DOUBLE_MAX);
	for (unsigned int i = 0; i < input_len; i++) {
		if (input[i] > m) {
			m = input[i];
		}
	}
	double sum = 0.0;
	for (unsigned int i = 0; i < input_len; i++)
		sum += pow(EULER_CONST, input[i] - m);
	double offset = m + pow(EULER_CONST, sum);
	for (unsigned int i = 0; i < input_len; i++)
		input[i] = pow(EULER_CONST, input[i] - offset);
}
