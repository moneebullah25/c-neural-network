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

	scanf_s("%d");
}