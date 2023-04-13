#include "neural_net.h"

int main()
{
	{
		/* Basic Perceptron Implementation */

		ANN* ann = ANNNew(2, 2, 1, 2);
		double inputs[] = { .05, .10 };
		double outputs[] = { .01, .99 };
		ANNUpdateWeights(ann, (double[]) { .15, .2, .25, .3, .4, .45, .5, .55 }, (double[]) { .35, .60 });

		double* total_error = malloc(sizeof(double));

		unsigned int epochs = 250;
		for (unsigned int i = 0; i < epochs; i++) {
			ANNForwardPropagate(ann, inputs, outputs, "SIGMOID", 0, total_error);
			ANNBackwardPropagate(ann, inputs, outputs, 0.5, "D_SIGMOID");

			printf("Epoch %u/%u\n", i, epochs);
			printf("1/1 [===============================] - loss: %.9f\n", *total_error);
		}
	}
	
	{
		/* Boston Housing Data */

		ANN* ann = ANNNew(13, 128, 1, 1);
		ANNRandomWeights(ann, 0., 1.);
		unsigned int nrows, ncols;

		double** dataset = ANNReadCSV("Z:/c-ann-matrix/c-ann-matrix/tests/BostonHousing.txt", &nrows, &ncols);
		double** train_feature, ** test_feature, * train_label, * test_label;

		// Calculate the number of rows in the training set
		unsigned int num_train_rows = (unsigned int)(nrows * 0.7);

		train_feature = (double**)malloc(num_train_rows * sizeof(double*));
		test_feature = (double**)malloc((nrows - num_train_rows) * sizeof(double*));

		ANNTrainTestSplit(dataset, nrows, ncols, 0.7, train_feature, test_feature);

		//PrintDataset(train_feature, num_train_rows, ncols);
		//PrintDataset(test_feature, nrows - num_train_rows, ncols);

		train_label = (double*)malloc(sizeof(double) * num_train_rows);
		test_label = (double*)malloc(sizeof(double) * nrows - num_train_rows);
		ANNLabelExtract(train_feature, 13, num_train_rows, ncols, train_label);
		ANNLabelExtract(test_feature, 13, nrows - num_train_rows, ncols, test_label);

		//PrintDataset(train_feature, num_train_rows, ncols);
		//PrintDatasetColumn(train_label, num_train_rows);
		//PrintDataset(test_feature, nrows - num_train_rows, ncols);
		//PrintDatasetColumn(test_label, nrows - num_train_rows);

		ANNDeleteFeature(train_feature, 13, num_train_rows, ncols);
		ANNDeleteFeature(test_feature, 13, nrows - num_train_rows, ncols);

		//PrintDataset(train_feature, num_train_rows, ncols);
		//PrintDataset(test_feature, nrows - num_train_rows, ncols - 1);

		unsigned int epochs = 250;
		double total_error;
		for (unsigned int epoch = 1; epoch <= epochs; epoch++)
		{
			for (unsigned int i = 0; i < num_train_rows; i++)
			{
				ANNTrain(ann, train_feature[i], &train_label[i], 0, 0.0001, "ReLU", "DReLU", &total_error);
			}
			printf("Epoch %u/%u\n", epoch, epochs);
			printf("1/1 [===============================] - loss: %.9f\n", total_error);
		}
	}

	int result;
	scanf_s("%d", &result);
}