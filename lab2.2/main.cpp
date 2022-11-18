#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>
#include <windows.h>

using namespace std;

typedef double(*TestFunctTempl)(double**&, double**&, double**&, int&, int&, int&);

double fillMatrix(double**& matrix, int size1, int size2) {

	double t_start = omp_get_wtime();
	for (int i = 0; i < size1; i++)
		for (int j = 0; j < size2; j++)
			matrix[i][j] = sin(i + 0.5) + cos(i / 2);
	double t_end = omp_get_wtime();
	return t_end - t_start;
}

double fillMatrixZero(double**& matrix, int size1, int size2)
{
	double time_start = omp_get_wtime();
	for (int i = 0; i < size1; i++) {
		for (int j = 0; j < size2; j++) {
			matrix[i][j] = 0;
		}
	}
	double time_stop = omp_get_wtime();
	return time_stop - time_start;
}

double fillMatrixParallelStatic(double**& matrix, int size1, int size2) {

	double t_start = omp_get_wtime();

#pragma omp parallel for schedule(static)
	for (int i = 0; i < size1; i++)
		for (int j = 0; j < size2; j++)
			matrix[i][j] = sin(i + 0.5) + cos(i / 2);

	double t_end = omp_get_wtime();
	return t_end - t_start;

}

double fillMatrixParallelDynamic(double**& matrix, int size1, int size2) {

	double t_start = omp_get_wtime();

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < size1; i++)
		for (int j = 0; j < size2; j++)
			matrix[i][j] = sin(i + 0.5) + cos(i / 2);

	double t_end = omp_get_wtime();
	return t_end - t_start;

}

double fillMatrixParallelGuided(double**& matrix, int size1, int size2) {

	double t_start = omp_get_wtime();

#pragma omp parallel for schedule(guided)
	for (int i = 0; i < size1; i++)
		for (int j = 0; j < size2; j++)
			matrix[i][j] = sin(i + 0.5) + cos(i / 2);

	double t_end = omp_get_wtime();
	return t_end - t_start;

}

double TestfillMatrix(double**& matrix1, double**& empty, double**& empty1, int& sizeA, int& sizeB, int& empty2)
{
	return fillMatrix(matrix1, sizeA, sizeB);
}
double TestfillMatrixParallelStatic(double**& empty, double**& matrix2, double**& empty1, int& sizeB, int& sizeA, int& empty2)
{
	return fillMatrixParallelGuided(matrix2, sizeA, sizeB);
}
double TestfillMatrixParallelDynamic(double**& matrix1, double**& empty, double**& empty1, int& sizeA, int& sizeB, int& empty2)
{
	return fillMatrixParallelGuided(matrix1, sizeA, sizeB);
}
double TestfillMatrixParallelGuided(double**& empty, double**& matrix2, double**& empty1, int& sizeB, int& sizeA, int& empty2)
{
	return fillMatrixParallelGuided(matrix2, sizeA, sizeB);
}

double multiplyMatrixV4(double**& matrix1, double**& matrix2, double**& matrix3, int& sizeA, int& sizeB, int& sizeC) {
	double time_start = omp_get_wtime();
	double** mtr = new double* [sizeC];
	for (int i = 0; i < sizeC; i++)
		mtr[i] = new double[sizeB];

	for (int i = 0; i < sizeB; i++)
		for (int j = 0; j < sizeC; j++)
			mtr[j][i] = matrix2[i][j];

	for (int i = 0; i < sizeA; i++) {
		for (int j = 0; j < sizeC; j++) {
			double tmp = 0;
			for (int k = 0; k < sizeB; k++) {
				tmp += matrix1[i][k] * mtr[j][k];
			}
			matrix3[i][j] = tmp;
		}
	}

	for (int i = 0; i < sizeC; i++)
	{
		delete[] mtr[i];
	}
	delete[] mtr;


	double time_stop = omp_get_wtime();
	return time_stop - time_start;
}

double multiplyMatrixV4ParallelStatic(double**& matrix1, double**& matrix2, double**& matrix3, int& sizeA, int& sizeB, int& sizeC) {
	double time_start = omp_get_wtime();
	double** mtr = new double* [sizeC];
	for (int i = 0; i < sizeC; i++)
		mtr[i] = new double[sizeB];

#pragma omp parallel for schedule(static, sizeB/100)
	for (int i = 0; i < sizeB; i++)
		for (int j = 0; j < sizeC; j++)
			mtr[j][i] = matrix2[i][j];
#pragma omp parallel for schedule(static, sizeA/100)
	for (int i = 0; i < sizeA; i++) {
		for (int j = 0; j < sizeC; j++) {
			double tmp = 0;
			for (int k = 0; k < sizeB; k++) {
				tmp += matrix1[i][k] * mtr[j][k];
			}
			matrix3[i][j] = tmp;
		}
	}

	for (int i = 0; i < sizeC; i++)
	{
		delete mtr[i];
	}
	delete[] mtr;

	double time_stop = omp_get_wtime();
	return time_stop - time_start;
}

double multiplyMatrixV4ParallelDynamic(double**& matrix1, double**& matrix2, double**& matrix3, int& sizeA, int& sizeB, int& sizeC) {
	double time_start = omp_get_wtime();
	double** mtr = new double* [sizeC];
	for (int i = 0; i < sizeC; i++)
		mtr[i] = new double[sizeB];

#pragma omp parallel for schedule(dynamic, sizeB/100)
	for (int i = 0; i < sizeB; i++)
		for (int j = 0; j < sizeC; j++)
			mtr[j][i] = matrix2[i][j];
#pragma omp parallel for schedule(dynamic, sizeA/100)
	for (int i = 0; i < sizeA; i++) {
		for (int j = 0; j < sizeC; j++) {
			double tmp = 0;
			for (int k = 0; k < sizeB; k++) {
				tmp += matrix1[i][k] * mtr[j][k];
			}
			matrix3[i][j] = tmp;
		}
	}

	for (int i = 0; i < sizeC; i++)
	{
		delete mtr[i];
	}
	delete[] mtr;

	double time_stop = omp_get_wtime();
	return time_stop - time_start;
}

double multiplyMatrixV4ParallelGuided(double**& matrix1, double**& matrix2, double**& matrix3, int& sizeA, int& sizeB, int& sizeC) {
	double time_start = omp_get_wtime();
	double** mtr = new double* [sizeC];
	for (int i = 0; i < sizeC; i++)
		mtr[i] = new double[sizeB];

#pragma omp parallel for schedule(guided, sizeB/100)
	for (int i = 0; i < sizeB; i++)
		for (int j = 0; j < sizeC; j++)
			mtr[j][i] = matrix2[i][j];
#pragma omp parallel for schedule(guided, sizeA/100)
	for (int i = 0; i < sizeA; i++) {
		for (int j = 0; j < sizeC; j++) {
			double tmp = 0;
			for (int k = 0; k < sizeB; k++) {
				tmp += matrix1[i][k] * mtr[j][k];
			}
			matrix3[i][j] = tmp;
		}
	}

	for (int i = 0; i < sizeC; i++)
	{
		delete mtr[i];
	}
	delete[] mtr;

	double time_stop = omp_get_wtime();
	return time_stop - time_start;
}

double TestmultiplyMatrixV4(double**& matrix1, double**& matrix2, double**& matrix3, int& sizeA, int& sizeB, int& sizeC)
{
	return multiplyMatrixV4(matrix1, matrix2, matrix3, sizeA, sizeB, sizeC);
}
double TestmultiplyMatrixV4ParallelStatic(double**& matrix1, double**& matrix2, double**& matrix3, int& sizeA, int& sizeB, int& sizeC)
{
	return multiplyMatrixV4ParallelStatic(matrix1, matrix2, matrix3, sizeA, sizeB, sizeC);
}
double TestmultiplyMatrixV4ParallelDynamic(double**& matrix1, double**& matrix2, double**& matrix3, int& sizeA, int& sizeB, int& sizeC)
{
	return multiplyMatrixV4ParallelDynamic(matrix1, matrix2, matrix3, sizeA, sizeB, sizeC);
}
double TestmultiplyMatrixV4ParallelGuided(double**& matrix1, double**& matrix2, double**& matrix3, int& sizeA, int& sizeB, int& sizeC)
{
	return multiplyMatrixV4ParallelGuided(matrix1, matrix2, matrix3, sizeA, sizeB, sizeC);
}

int validateMatrix(double**& matrix1, double**& matrix2, int size1, int size2)
{
	int flag = 1;
	for (int i = 0; i < size1; i++) {
		for (int j = 0; j < size2; j++) {
			if (float(matrix1[i][j]) != float(matrix2[i][j]))
			{
				std::cout << "Mtx:" << i << "x" << j << " " << matrix1[i][j] << " " << matrix2[i][j] << std::endl;
				flag = 0;
				break;
			}
			if (flag == 0)
				break;
		}

	}
	return flag;
}

int ADD(double**& MatrixA, double**& MatrixB, double**& MatrixResult, int MatrixSize)
{
	for (int i = 0; i < MatrixSize; i++)
	{
		for (int j = 0; j < MatrixSize; j++)
		{
			MatrixResult[i][j] = MatrixA[i][j] + MatrixB[i][j];
		}
	}
	return 0;
}

int SUB(double**& MatrixA, double**& MatrixB, double**& MatrixResult, int MatrixSize)
{
	for (int i = 0; i < MatrixSize; i++)
	{
		for (int j = 0; j < MatrixSize; j++)
		{
			MatrixResult[i][j] = MatrixA[i][j] - MatrixB[i][j];
		}
	}
	return 0;
}

int MUL(double** MatrixA, double** MatrixB, double** MatrixResult, int MatrixSize)
{
	double** mtr = new double* [MatrixSize];
	mtr[0] = new double[MatrixSize * MatrixSize];
	for (int i = 1; i < MatrixSize; i++)
		mtr[i] = &mtr[0][i * MatrixSize];
	for (int i = 0; i < MatrixSize; i++)
		for (int j = 0; j < MatrixSize; j++)
			mtr[j][i] = MatrixB[i][j];
	for (int i = 0; i < MatrixSize; i++) {
		for (int j = 0; j < MatrixSize; j++) {
			double tmp = 0;
			for (int k = 0; k < MatrixSize; k++) {
				tmp += MatrixA[i][k] * mtr[j][k];
			}
			MatrixResult[i][j] = tmp;
		}
	}
	delete[] mtr[0];
	delete[] mtr;

	return 0;
}


int ADDParallelGuided(double**& MatrixA, double**& MatrixB, double**& MatrixResult, int MatrixSize)
{
#pragma omp parallel for schedule(guided, MatrixSize/10)
	for (int i = 0; i < MatrixSize; i++)
	{
		for (int j = 0; j < MatrixSize; j++)
		{
			MatrixResult[i][j] = MatrixA[i][j] + MatrixB[i][j];
		}
	}
	return 0;
}

int SUBParallelGuided(double**& MatrixA, double**& MatrixB, double**& MatrixResult, int MatrixSize)
{
#pragma omp parallel for schedule(guided, MatrixSize/10)
	for (int i = 0; i < MatrixSize; i++)
	{
		for (int j = 0; j < MatrixSize; j++)
		{
			MatrixResult[i][j] = MatrixA[i][j] - MatrixB[i][j];
		}
	}
	return 0;
}

int MULParallelGuided(double** MatrixA, double** MatrixB, double** MatrixResult, int MatrixSize)
{
	double** mtr = new double* [MatrixSize];
	mtr[0] = new double[MatrixSize * MatrixSize];
	for (int i = 1; i < MatrixSize; i++)
		mtr[i] = &mtr[0][i * MatrixSize];

#pragma omp parallel for schedule(guided, MatrixSize/10)
	for (int i = 0; i < MatrixSize; i++)
		for (int j = 0; j < MatrixSize; j++)
			mtr[j][i] = MatrixB[i][j];
#pragma omp parallel for schedule(guided, MatrixSize/10)
	for (int i = 0; i < MatrixSize; i++) {
		for (int j = 0; j < MatrixSize; j++) {
			double tmp = 0;
			for (int k = 0; k < MatrixSize; k++) {
				tmp += MatrixA[i][k] * mtr[j][k];
			}
			MatrixResult[i][j] = tmp;
		}
	}

	delete[] mtr[0];
	delete[] mtr;

	return 0;
}


int ADDParallelSections(double**& MatrixA, double**& MatrixB, double**& MatrixResult, int MatrixSize)
{
	int n_t = 0;
#pragma omp parallel 
	{
		n_t = omp_get_num_threads();
	}
	int st1 = MatrixSize / n_t;
	int st2 = MatrixSize * 2 / n_t;
	int st3 = MatrixSize * 3 / n_t;

#pragma omp parallel sections 
	{
#pragma omp section
		{
			for (int i = 0; i < st1; i++)
			{
				for (int j = 0; j < st1; j++)
				{
					MatrixResult[i][j] = MatrixA[i][j] + MatrixB[i][j];
				}
			}
		}
#pragma omp section
		{
			if (n_t > 1)
			{
				for (int i = st1; i < st2; i++)
				{
					for (int j = st1; j < st2; j++)
					{
						MatrixResult[i][j] = MatrixA[i][j] + MatrixB[i][j];
					}
				}
			}
		}
#pragma omp section
		{
			if (n_t > 2)
			{
				for (int i = st2; i < st3; i++)
				{
					for (int j = st2; j < st3; j++)
					{
						MatrixResult[i][j] = MatrixA[i][j] + MatrixB[i][j];
					}
				}
			}
		}
#pragma omp section
		{
			if (n_t > 3)
			{
				for (int i = st3; i < MatrixSize; i++)
				{
					for (int j = st3; j < MatrixSize; j++)
					{
						MatrixResult[i][j] = MatrixA[i][j] + MatrixB[i][j];
					}
				}
			}
		}
	}
	return 0;
}

int SUBParallelSections(double**& MatrixA, double**& MatrixB, double**& MatrixResult, int MatrixSize)
{
	int n_t = 0;
#pragma omp parallel 
	{
		n_t = omp_get_num_threads();
	}
	int st1 = MatrixSize / n_t;
	int st2 = MatrixSize * 2 / n_t;
	int st3 = MatrixSize * 3 / n_t;

#pragma omp parallel sections 
	{
#pragma omp section
		{
			for (int i = 0; i < st1; i++)
			{
				for (int j = 0; j < st1; j++)
				{
					MatrixResult[i][j] = MatrixA[i][j] - MatrixB[i][j];
				}
			}
		}
#pragma omp section
		{
			if (n_t > 1)
			{
				for (int i = st1; i < st2; i++)
				{
					for (int j = st1; j < st2; j++)
					{
						MatrixResult[i][j] = MatrixA[i][j] - MatrixB[i][j];
					}
				}
			}
		}
#pragma omp section
		{
			if (n_t > 2)
			{
				for (int i = st2; i < st3; i++)
				{
					for (int j = st2; j < st3; j++)
					{
						MatrixResult[i][j] = MatrixA[i][j] - MatrixB[i][j];
					}
				}
			}
		}
#pragma omp section
		{
			if (n_t > 3)
			{
				for (int i = st3; i < MatrixSize; i++)
				{
					for (int j = st3; j < MatrixSize; j++)
					{
						MatrixResult[i][j] = MatrixA[i][j] - MatrixB[i][j];
					}
				}
			}
		}
	}

	return 0;
}

int MULParallelSections(double** MatrixA, double** MatrixB, double** MatrixResult, int MatrixSize)
{
	double** mtr = new double* [MatrixSize];
	mtr[0] = new double[MatrixSize * MatrixSize];
	for (int i = 1; i < MatrixSize; i++)
		mtr[i] = &mtr[0][i * MatrixSize];

	for (int i = 0; i < MatrixSize; i++)
		for (int j = 0; j < MatrixSize; j++)
			mtr[j][i] = MatrixB[i][j];

	int n_t = 0;

#pragma omp parallel 
	{
		n_t = omp_get_num_threads();
	}
	int st1 = MatrixSize / n_t;
	int st2 = MatrixSize * 2 / n_t;
	int st3 = MatrixSize * 3 / n_t;

#pragma omp parallel sections 
	{
#pragma omp section
		{
			for (int i = 0; i < st1; i++) {
				for (int j = 0; j < st1; j++) {
					double tmp = 0;
					for (int k = 0; k < st1; k++) {
						tmp += MatrixA[i][k] * mtr[j][k];
					}
					MatrixResult[i][j] = tmp;
				}
			}
		}
#pragma omp section
		{
			if (n_t > 1)
			{
				for (int i = st1; i < st2; i++) {
					for (int j = st1; j < st2; j++) {
						double tmp = 0;
						for (int k = st1; k < st2; k++) {
							tmp += MatrixA[i][k] * mtr[j][k];
						}
						MatrixResult[i][j] = tmp;
					}
				}
			}
		}
#pragma omp section
		{
			if (n_t > 2)
			{
				for (int i = st2; i < st3; i++) {
					for (int j = st2; j < st3; j++) {
						double tmp = 0;
						for (int k = st2; k < st3; k++) {
							tmp += MatrixA[i][k] * mtr[j][k];
						}
						MatrixResult[i][j] = tmp;
					}
				}
			}
		}
#pragma omp section
		{
			if (n_t > 3)
			{
				for (int i = st3; i < MatrixSize; i++) {
					for (int j = st3; j < MatrixSize; j++) {
						double tmp = 0;
						for (int k = st3; k < MatrixSize; k++) {
							tmp += MatrixA[i][k] * mtr[j][k];
						}
						MatrixResult[i][j] = tmp;
					}
				}
			}
		}
	}
	delete[] mtr[0];
	delete[] mtr;

	return 0;
}

int Strassen(double** MatrixA, double** MatrixB, double** MatrixC, int MatrixSize, int linearMultBlockSize)
{
	int HalfSize = MatrixSize / 2;

	if (MatrixSize <= linearMultBlockSize)
	{
		MUL(MatrixA, MatrixB, MatrixC, MatrixSize);
	}
	else
	{
		double** A11, ** A12, ** A21, ** A22;
		double** B11, ** B12, ** B21, ** B22;
		double** C11, ** C12, ** C21, ** C22;
		double** M1, ** M2, ** M3, ** M4, ** M5, ** M6, ** M7;
		double** AResult, ** BResult;

		A11 = new double* [HalfSize];
		A12 = new double* [HalfSize];
		A21 = new double* [HalfSize];
		A22 = new double* [HalfSize];

		B11 = new double* [HalfSize];
		B12 = new double* [HalfSize];
		B21 = new double* [HalfSize];
		B22 = new double* [HalfSize];

		C11 = new double* [HalfSize];
		C12 = new double* [HalfSize];
		C21 = new double* [HalfSize];
		C22 = new double* [HalfSize];

		M1 = new double* [HalfSize];
		M2 = new double* [HalfSize];
		M3 = new double* [HalfSize];
		M4 = new double* [HalfSize];
		M5 = new double* [HalfSize];
		M6 = new double* [HalfSize];
		M7 = new double* [HalfSize];

		AResult = new double* [HalfSize];
		BResult = new double* [HalfSize];

		A11[0] = new double[HalfSize * HalfSize];
		A12[0] = new double[HalfSize * HalfSize];
		A21[0] = new double[HalfSize * HalfSize];
		A22[0] = new double[HalfSize * HalfSize];

		B11[0] = new double[HalfSize * HalfSize];
		B12[0] = new double[HalfSize * HalfSize];
		B21[0] = new double[HalfSize * HalfSize];
		B22[0] = new double[HalfSize * HalfSize];

		C11[0] = new double[HalfSize * HalfSize];
		C12[0] = new double[HalfSize * HalfSize];
		C21[0] = new double[HalfSize * HalfSize];
		C22[0] = new double[HalfSize * HalfSize];

		M1[0] = new double[HalfSize * HalfSize];
		M2[0] = new double[HalfSize * HalfSize];
		M3[0] = new double[HalfSize * HalfSize];
		M4[0] = new double[HalfSize * HalfSize];
		M5[0] = new double[HalfSize * HalfSize];
		M6[0] = new double[HalfSize * HalfSize];
		M7[0] = new double[HalfSize * HalfSize];

		AResult[0] = new double[HalfSize * HalfSize];
		BResult[0] = new double[HalfSize * HalfSize];

		for (int i = 0; i < HalfSize; i++)
		{
			A11[i] = &A11[0][i * HalfSize];
			A12[i] = &A12[0][i * HalfSize];
			A21[i] = &A21[0][i * HalfSize];
			A22[i] = &A22[0][i * HalfSize];

			B11[i] = &B11[0][i * HalfSize];
			B12[i] = &B12[0][i * HalfSize];
			B21[i] = &B21[0][i * HalfSize];
			B22[i] = &B22[0][i * HalfSize];

			C11[i] = &C11[0][i * HalfSize];
			C12[i] = &C12[0][i * HalfSize];
			C21[i] = &C21[0][i * HalfSize];
			C22[i] = &C22[0][i * HalfSize];

			M1[i] = &M1[0][i * HalfSize];
			M2[i] = &M2[0][i * HalfSize];
			M3[i] = &M3[0][i * HalfSize];
			M4[i] = &M4[0][i * HalfSize];
			M5[i] = &M5[0][i * HalfSize];
			M6[i] = &M6[0][i * HalfSize];
			M7[i] = &M7[0][i * HalfSize];

			AResult[i] = &AResult[0][i * HalfSize];
			BResult[i] = &BResult[0][i * HalfSize];
		}
		/////////////////////////////////////////

		for (int i = 0; i < HalfSize; i++)
		{
			for (int j = 0; j < HalfSize; j++)
			{
				A11[i][j] = MatrixA[i][j];
				A12[i][j] = MatrixA[i][j + HalfSize];
				A21[i][j] = MatrixA[i + HalfSize][j];
				A22[i][j] = MatrixA[i + HalfSize][j + HalfSize];

				B11[i][j] = MatrixB[i][j];
				B12[i][j] = MatrixB[i][j + HalfSize];
				B21[i][j] = MatrixB[i + HalfSize][j];
				B22[i][j] = MatrixB[i + HalfSize][j + HalfSize];

			}
		}

		//P1 == M1[][]
		ADD(A11, A22, AResult, HalfSize);
		ADD(B11, B22, BResult, HalfSize);
		Strassen(AResult, BResult, M1, HalfSize, linearMultBlockSize);


		//P2 == M2[][]
		ADD(A21, A22, AResult, HalfSize);              //M2=(A21+A22)B11
		Strassen(AResult, B11, M2, HalfSize, linearMultBlockSize);       //Mul(AResult,B11,M2);

		//P3 == M3[][]
		SUB(B12, B22, BResult, HalfSize);              //M3=A11(B12-B22)
		Strassen(A11, BResult, M3, HalfSize, linearMultBlockSize);       //Mul(A11,BResult,M3);

		//P4 == M4[][]
		SUB(B21, B11, BResult, HalfSize);           //M4=A22(B21-B11)
		Strassen(A22, BResult, M4, HalfSize, linearMultBlockSize);       //Mul(A22,BResult,M4);

		//P5 == M5[][]
		ADD(A11, A12, AResult, HalfSize);           //M5=(A11+A12)B22
		Strassen(AResult, B22, M5, HalfSize, linearMultBlockSize);       //Mul(AResult,B22,M5);


		//P6 == M6[][]
		SUB(A21, A11, AResult, HalfSize);
		ADD(B11, B12, BResult, HalfSize);             //M6=(A21-A11)(B11+B12)
		Strassen(AResult, BResult, M6, HalfSize, linearMultBlockSize);    //Mul(AResult,BResult,M6);

		//P7 == M7[][]
		SUB(A12, A22, AResult, HalfSize);
		ADD(B21, B22, BResult, HalfSize);             //M7=(A12-A22)(B21+B22)
		Strassen(AResult, BResult, M7, HalfSize, linearMultBlockSize);     //Mul(AResult,BResult,M7);

		//C11 = M1 + M4 - M5 + M7;
		ADD(M1, M4, AResult, HalfSize);
		SUB(M7, M5, BResult, HalfSize);
		ADD(AResult, BResult, C11, HalfSize);

		//C12 = M3 + M5;
		ADD(M3, M5, C12, HalfSize);

		//C21 = M2 + M4;
		ADD(M2, M4, C21, HalfSize);

		//C22 = M1 + M3 - M2 + M6;
		ADD(M1, M3, AResult, HalfSize);
		SUB(M6, M2, BResult, HalfSize);
		ADD(AResult, BResult, C22, HalfSize);


		for (int i = 0; i < HalfSize; i++)
		{
			for (int j = 0; j < HalfSize; j++)
			{
				MatrixC[i][j] = C11[i][j];
				MatrixC[i][j + HalfSize] = C12[i][j];
				MatrixC[i + HalfSize][j] = C21[i][j];
				MatrixC[i + HalfSize][j + HalfSize] = C22[i][j];
			}
		}

		delete[] A11[0]; delete[] A12[0]; delete[] A21[0]; delete[] A22[0];
		delete[] B11[0]; delete[] B12[0]; delete[] B21[0]; delete[] B22[0];
		delete[] C11[0]; delete[] C12[0]; delete[] C21[0]; delete[] C22[0];
		delete[] M1[0]; delete[] M2[0]; delete[] M3[0]; delete[] M4[0]; delete[] M5[0];
		delete[] M6[0]; delete[] M7[0];
		delete[] AResult[0];
		delete[] BResult[0];

		delete[] A11; delete[] A12; delete[] A21; delete[] A22;
		delete[] B11; delete[] B12; delete[] B21; delete[] B22;
		delete[] C11; delete[] C12; delete[] C21; delete[] C22;
		delete[] M1; delete[] M2; delete[] M3; delete[] M4; delete[] M5;
		delete[] M6; delete[] M7;
		delete[] AResult;
		delete[] BResult;


	}
	return 0;
}
int StrassenParallelGuided(double** MatrixA, double** MatrixB, double** MatrixC, int MatrixSize, int linearMultBlockSize)
{
	int HalfSize = MatrixSize / 2;

	if (MatrixSize <= linearMultBlockSize)
	{
		MULParallelGuided(MatrixA, MatrixB, MatrixC, MatrixSize);
	}
	else
	{
		double** A11, ** A12, ** A21, ** A22;
		double** B11, ** B12, ** B21, ** B22;
		double** C11, ** C12, ** C21, ** C22;
		double** M1, ** M2, ** M3, ** M4, ** M5, ** M6, ** M7;
		double** AResult, ** BResult;

		A11 = new double* [HalfSize];
		A12 = new double* [HalfSize];
		A21 = new double* [HalfSize];
		A22 = new double* [HalfSize];

		B11 = new double* [HalfSize];
		B12 = new double* [HalfSize];
		B21 = new double* [HalfSize];
		B22 = new double* [HalfSize];

		C11 = new double* [HalfSize];
		C12 = new double* [HalfSize];
		C21 = new double* [HalfSize];
		C22 = new double* [HalfSize];

		M1 = new double* [HalfSize];
		M2 = new double* [HalfSize];
		M3 = new double* [HalfSize];
		M4 = new double* [HalfSize];
		M5 = new double* [HalfSize];
		M6 = new double* [HalfSize];
		M7 = new double* [HalfSize];

		AResult = new double* [HalfSize];
		BResult = new double* [HalfSize];

		A11[0] = new double[HalfSize * HalfSize];
		A12[0] = new double[HalfSize * HalfSize];
		A21[0] = new double[HalfSize * HalfSize];
		A22[0] = new double[HalfSize * HalfSize];

		B11[0] = new double[HalfSize * HalfSize];
		B12[0] = new double[HalfSize * HalfSize];
		B21[0] = new double[HalfSize * HalfSize];
		B22[0] = new double[HalfSize * HalfSize];

		C11[0] = new double[HalfSize * HalfSize];
		C12[0] = new double[HalfSize * HalfSize];
		C21[0] = new double[HalfSize * HalfSize];
		C22[0] = new double[HalfSize * HalfSize];

		M1[0] = new double[HalfSize * HalfSize];
		M2[0] = new double[HalfSize * HalfSize];
		M3[0] = new double[HalfSize * HalfSize];
		M4[0] = new double[HalfSize * HalfSize];
		M5[0] = new double[HalfSize * HalfSize];
		M6[0] = new double[HalfSize * HalfSize];
		M7[0] = new double[HalfSize * HalfSize];

		AResult[0] = new double[HalfSize * HalfSize];
		BResult[0] = new double[HalfSize * HalfSize];

#pragma omp parallel for schedule(guided, HalfSize/10)
		for (int i = 0; i < HalfSize; i++)
		{
			A11[i] = &A11[0][i * HalfSize];
			A12[i] = &A12[0][i * HalfSize];
			A21[i] = &A21[0][i * HalfSize];
			A22[i] = &A22[0][i * HalfSize];

			B11[i] = &B11[0][i * HalfSize];
			B12[i] = &B12[0][i * HalfSize];
			B21[i] = &B21[0][i * HalfSize];
			B22[i] = &B22[0][i * HalfSize];

			C11[i] = &C11[0][i * HalfSize];
			C12[i] = &C12[0][i * HalfSize];
			C21[i] = &C21[0][i * HalfSize];
			C22[i] = &C22[0][i * HalfSize];

			M1[i] = &M1[0][i * HalfSize];
			M2[i] = &M2[0][i * HalfSize];
			M3[i] = &M3[0][i * HalfSize];
			M4[i] = &M4[0][i * HalfSize];
			M5[i] = &M5[0][i * HalfSize];
			M6[i] = &M6[0][i * HalfSize];
			M7[i] = &M7[0][i * HalfSize];

			AResult[i] = &AResult[0][i * HalfSize];
			BResult[i] = &BResult[0][i * HalfSize];
		}
		/////////////////////////////////////////
#pragma omp parallel for schedule(guided, HalfSize/10)
		for (int i = 0; i < HalfSize; i++)
		{
			for (int j = 0; j < HalfSize; j++)
			{
				A11[i][j] = MatrixA[i][j];
				A12[i][j] = MatrixA[i][j + HalfSize];
				A21[i][j] = MatrixA[i + HalfSize][j];
				A22[i][j] = MatrixA[i + HalfSize][j + HalfSize];

				B11[i][j] = MatrixB[i][j];
				B12[i][j] = MatrixB[i][j + HalfSize];
				B21[i][j] = MatrixB[i + HalfSize][j];
				B22[i][j] = MatrixB[i + HalfSize][j + HalfSize];

			}
		}
		//P1 == M1[][]
		ADDParallelGuided(A11, A22, AResult, HalfSize);
		ADDParallelGuided(B11, B22, BResult, HalfSize);
		StrassenParallelGuided(AResult, BResult, M1, HalfSize, linearMultBlockSize);

		//P2 == M2[][]
		ADDParallelGuided(A21, A22, AResult, HalfSize);              //M2=(A21+A22)B11
		StrassenParallelGuided(AResult, B11, M2, HalfSize, linearMultBlockSize);       //Mul(AResult,B11,M2);

		//P3 == M3[][]
		SUBParallelGuided(B12, B22, BResult, HalfSize);              //M3=A11(B12-B22)
		StrassenParallelGuided(A11, BResult, M3, HalfSize, linearMultBlockSize);       //Mul(A11,BResult,M3);

		//P4 == M4[][]
		SUBParallelGuided(B21, B11, BResult, HalfSize);           //M4=A22(B21-B11)
		StrassenParallelGuided(A22, BResult, M4, HalfSize, linearMultBlockSize);       //Mul(A22,BResult,M4);

		//P5 == M5[][]
		ADDParallelGuided(A11, A12, AResult, HalfSize);           //M5=(A11+A12)B22
		StrassenParallelGuided(AResult, B22, M5, HalfSize, linearMultBlockSize);       //Mul(AResult,B22,M5);

		//P6 == M6[][]
		SUBParallelGuided(A21, A11, AResult, HalfSize);
		ADDParallelGuided(B11, B12, BResult, HalfSize);             //M6=(A21-A11)(B11+B12)
		StrassenParallelGuided(AResult, BResult, M6, HalfSize, linearMultBlockSize);    //Mul(AResult,BResult,M6);

		//P7 == M7[][]
		SUBParallelGuided(A12, A22, AResult, HalfSize);
		ADDParallelGuided(B21, B22, BResult, HalfSize);             //M7=(A12-A22)(B21+B22)
		StrassenParallelGuided(AResult, BResult, M7, HalfSize, linearMultBlockSize);     //Mul(AResult,BResult,M7);

		//C11 = M1 + M4 - M5 + M7;
		ADDParallelGuided(M1, M4, AResult, HalfSize);
		SUBParallelGuided(M7, M5, BResult, HalfSize);
		ADDParallelGuided(AResult, BResult, C11, HalfSize);

		//C12 = M3 + M5;
		ADDParallelGuided(M3, M5, C12, HalfSize);

		//C21 = M2 + M4;
		ADDParallelGuided(M2, M4, C21, HalfSize);

		//C22 = M1 + M3 - M2 + M6;
		ADDParallelGuided(M1, M3, AResult, HalfSize);
		SUBParallelGuided(M6, M2, BResult, HalfSize);
		ADDParallelGuided(AResult, BResult, C22, HalfSize);

#pragma omp parallel for schedule(guided, HalfSize/10)
		for (int i = 0; i < HalfSize; i++)
		{
			for (int j = 0; j < HalfSize; j++)
			{
				MatrixC[i][j] = C11[i][j];
				MatrixC[i][j + HalfSize] = C12[i][j];
				MatrixC[i + HalfSize][j] = C21[i][j];
				MatrixC[i + HalfSize][j + HalfSize] = C22[i][j];
			}
		}

		delete[] A11[0]; delete[] A12[0]; delete[] A21[0]; delete[] A22[0];
		delete[] B11[0]; delete[] B12[0]; delete[] B21[0]; delete[] B22[0];
		delete[] C11[0]; delete[] C12[0]; delete[] C21[0]; delete[] C22[0];
		delete[] M1[0]; delete[] M2[0]; delete[] M3[0]; delete[] M4[0]; delete[] M5[0];
		delete[] M6[0]; delete[] M7[0];
		delete[] AResult[0];
		delete[] BResult[0];

		delete[] A11; delete[] A12; delete[] A21; delete[] A22;
		delete[] B11; delete[] B12; delete[] B21; delete[] B22;
		delete[] C11; delete[] C12; delete[] C21; delete[] C22;
		delete[] M1; delete[] M2; delete[] M3; delete[] M4; delete[] M5;
		delete[] M6; delete[] M7;
		delete[] AResult;
		delete[] BResult;


	}
	return 0;
}

int StrassenParallelSections(double** MatrixA, double** MatrixB, double** MatrixC, int MatrixSize, int linearMultBlockSize)
{
	int HalfSize = MatrixSize / 2;

	if (MatrixSize <= linearMultBlockSize)
	{
		MULParallelSections(MatrixA, MatrixB, MatrixC, MatrixSize);
	}
	else
	{
		double** A11, ** A12, ** A21, ** A22;
		double** B11, ** B12, ** B21, ** B22;
		double** C11, ** C12, ** C21, ** C22;
		double** M1, ** M2, ** M3, ** M4, ** M5, ** M6, ** M7;
		double** AResult, ** BResult;

		A11 = new double* [HalfSize];
		A12 = new double* [HalfSize];
		A21 = new double* [HalfSize];
		A22 = new double* [HalfSize];

		B11 = new double* [HalfSize];
		B12 = new double* [HalfSize];
		B21 = new double* [HalfSize];
		B22 = new double* [HalfSize];

		C11 = new double* [HalfSize];
		C12 = new double* [HalfSize];
		C21 = new double* [HalfSize];
		C22 = new double* [HalfSize];

		M1 = new double* [HalfSize];
		M2 = new double* [HalfSize];
		M3 = new double* [HalfSize];
		M4 = new double* [HalfSize];
		M5 = new double* [HalfSize];
		M6 = new double* [HalfSize];
		M7 = new double* [HalfSize];

		AResult = new double* [HalfSize];
		BResult = new double* [HalfSize];

		A11[0] = new double[HalfSize * HalfSize];
		A12[0] = new double[HalfSize * HalfSize];
		A21[0] = new double[HalfSize * HalfSize];
		A22[0] = new double[HalfSize * HalfSize];

		B11[0] = new double[HalfSize * HalfSize];
		B12[0] = new double[HalfSize * HalfSize];
		B21[0] = new double[HalfSize * HalfSize];
		B22[0] = new double[HalfSize * HalfSize];

		C11[0] = new double[HalfSize * HalfSize];
		C12[0] = new double[HalfSize * HalfSize];
		C21[0] = new double[HalfSize * HalfSize];
		C22[0] = new double[HalfSize * HalfSize];

		M1[0] = new double[HalfSize * HalfSize];
		M2[0] = new double[HalfSize * HalfSize];
		M3[0] = new double[HalfSize * HalfSize];
		M4[0] = new double[HalfSize * HalfSize];
		M5[0] = new double[HalfSize * HalfSize];
		M6[0] = new double[HalfSize * HalfSize];
		M7[0] = new double[HalfSize * HalfSize];

		AResult[0] = new double[HalfSize * HalfSize];
		BResult[0] = new double[HalfSize * HalfSize];

		for (int i = 0; i < HalfSize; i++)
		{
			A11[i] = &A11[0][i * HalfSize];
			A12[i] = &A12[0][i * HalfSize];
			A21[i] = &A21[0][i * HalfSize];
			A22[i] = &A22[0][i * HalfSize];

			B11[i] = &B11[0][i * HalfSize];
			B12[i] = &B12[0][i * HalfSize];
			B21[i] = &B21[0][i * HalfSize];
			B22[i] = &B22[0][i * HalfSize];

			C11[i] = &C11[0][i * HalfSize];
			C12[i] = &C12[0][i * HalfSize];
			C21[i] = &C21[0][i * HalfSize];
			C22[i] = &C22[0][i * HalfSize];

			M1[i] = &M1[0][i * HalfSize];
			M2[i] = &M2[0][i * HalfSize];
			M3[i] = &M3[0][i * HalfSize];
			M4[i] = &M4[0][i * HalfSize];
			M5[i] = &M5[0][i * HalfSize];
			M6[i] = &M6[0][i * HalfSize];
			M7[i] = &M7[0][i * HalfSize];

			AResult[i] = &AResult[0][i * HalfSize];
			BResult[i] = &BResult[0][i * HalfSize];
		}
		/////////////////////////////////////////

		for (int i = 0; i < HalfSize; i++)
		{
			for (int j = 0; j < HalfSize; j++)
			{
				A11[i][j] = MatrixA[i][j];
				A12[i][j] = MatrixA[i][j + HalfSize];
				A21[i][j] = MatrixA[i + HalfSize][j];
				A22[i][j] = MatrixA[i + HalfSize][j + HalfSize];

				B11[i][j] = MatrixB[i][j];
				B12[i][j] = MatrixB[i][j + HalfSize];
				B21[i][j] = MatrixB[i + HalfSize][j];
				B22[i][j] = MatrixB[i + HalfSize][j + HalfSize];

			}
		}
		//P1 == M1[][]
		ADDParallelSections(A11, A22, AResult, HalfSize);
		ADDParallelSections(B11, B22, BResult, HalfSize);
		StrassenParallelSections(AResult, BResult, M1, HalfSize, linearMultBlockSize);

		//P2 == M2[][]
		ADDParallelSections(A21, A22, AResult, HalfSize);              //M2=(A21+A22)B11
		StrassenParallelSections(AResult, B11, M2, HalfSize, linearMultBlockSize);       //Mul(AResult,B11,M2);

		//P3 == M3[][]
		SUBParallelSections(B12, B22, BResult, HalfSize);              //M3=A11(B12-B22)
		StrassenParallelSections(A11, BResult, M3, HalfSize, linearMultBlockSize);       //Mul(A11,BResult,M3);

		//P4 == M4[][]
		SUBParallelSections(B21, B11, BResult, HalfSize);           //M4=A22(B21-B11)
		StrassenParallelSections(A22, BResult, M4, HalfSize, linearMultBlockSize);       //Mul(A22,BResult,M4);

		//P5 == M5[][]
		ADDParallelSections(A11, A12, AResult, HalfSize);           //M5=(A11+A12)B22
		StrassenParallelSections(AResult, B22, M5, HalfSize, linearMultBlockSize);       //Mul(AResult,B22,M5);

		//P6 == M6[][]
		SUBParallelSections(A21, A11, AResult, HalfSize);
		ADDParallelSections(B11, B12, BResult, HalfSize);             //M6=(A21-A11)(B11+B12)
		StrassenParallelSections(AResult, BResult, M6, HalfSize, linearMultBlockSize);    //Mul(AResult,BResult,M6);

		//P7 == M7[][]
		SUBParallelSections(A12, A22, AResult, HalfSize);
		ADDParallelSections(B21, B22, BResult, HalfSize);             //M7=(A12-A22)(B21+B22)
		StrassenParallelSections(AResult, BResult, M7, HalfSize, linearMultBlockSize);     //Mul(AResult,BResult,M7);

		//C11 = M1 + M4 - M5 + M7;
		ADDParallelSections(M1, M4, AResult, HalfSize);
		SUBParallelSections(M7, M5, BResult, HalfSize);
		ADDParallelSections(AResult, BResult, C11, HalfSize);

		//C12 = M3 + M5;
		ADDParallelSections(M3, M5, C12, HalfSize);

		//C21 = M2 + M4;
		ADDParallelSections(M2, M4, C21, HalfSize);

		//C22 = M1 + M3 - M2 + M6;
		ADDParallelSections(M1, M3, AResult, HalfSize);
		SUBParallelSections(M6, M2, BResult, HalfSize);
		ADDParallelSections(AResult, BResult, C22, HalfSize);


		for (int i = 0; i < HalfSize; i++)
		{
			for (int j = 0; j < HalfSize; j++)
			{
				MatrixC[i][j] = C11[i][j];
				MatrixC[i][j + HalfSize] = C12[i][j];
				MatrixC[i + HalfSize][j] = C21[i][j];
				MatrixC[i + HalfSize][j + HalfSize] = C22[i][j];
			}
		}



		delete[] A11[0]; delete[] A12[0]; delete[] A21[0]; delete[] A22[0];
		delete[] B11[0]; delete[] B12[0]; delete[] B21[0]; delete[] B22[0];
		delete[] C11[0]; delete[] C12[0]; delete[] C21[0]; delete[] C22[0];
		delete[] M1[0]; delete[] M2[0]; delete[] M3[0]; delete[] M4[0]; delete[] M5[0];
		delete[] M6[0]; delete[] M7[0];
		delete[] AResult[0];
		delete[] BResult[0];

		delete[] A11; delete[] A12; delete[] A21; delete[] A22;
		delete[] B11; delete[] B12; delete[] B21; delete[] B22;
		delete[] C11; delete[] C12; delete[] C21; delete[] C22;
		delete[] M1; delete[] M2; delete[] M3; delete[] M4; delete[] M5;
		delete[] M6; delete[] M7;
		delete[] AResult;
		delete[] BResult;


	}
	return 0;
}

int Validate_size(int size, int linearMultBlockSize)
{
	int tms = size;
	while (tms > linearMultBlockSize)
	{
		if (tms % 2 == 0)
		{
			tms /= 2;
		}
		else
			return 0;
	}
	return 1;
}

int Find_Valid_Size(int size, int linearMultBlockSize)
{
	int newsize = size;
	while (Validate_size(newsize, linearMultBlockSize) == 0)
	{
		newsize++;
	}
	return newsize;
}

double** ExpandMatrixToSize(double** matrix, int sizeA, int sizeB, int NewSize)
{
	double** NewMatr = new double* [NewSize];
	for (int i = 0; i < NewSize; i++)
	{
		NewMatr[i] = new double[NewSize];
		for (int j = 0; j < NewSize; j++)
		{
			NewMatr[i][j] = 0;
		}
	}
	for (int i = 0; i < sizeA; i++)
		for (int j = 0; j < sizeB; j++)
		{
			NewMatr[i][j] = matrix[i][j];
		}
	return NewMatr;
}

double Shtrassen_Multiplication(double** A, double** B, double** C, int sizeA, int sizeB, int sizeC)
{
	int linearMultBlockSize = 64;
	double t_st = 0, t_ed = -1;
	int size = sizeA;
	double time = 0;

	if (sizeA == sizeB && sizeA == sizeC && Validate_size(size, linearMultBlockSize))
	{
		t_st = omp_get_wtime();
		Strassen(A, B, C, size, linearMultBlockSize);
		time = omp_get_wtime() - t_st;
		return time;
	}
	else
	{
		if (size < sizeB) size = sizeB;
		if (size < sizeC) size = sizeC;
		if (size != Find_Valid_Size(size, linearMultBlockSize))
		{
			size = Find_Valid_Size(size, linearMultBlockSize);
		}

		t_st = omp_get_wtime();
#pragma align(4)
		double** TA = ExpandMatrixToSize(A, sizeA, sizeB, size);
#pragma align(4)
		double** TB = ExpandMatrixToSize(B, sizeB, sizeC, size);
#pragma align(4)
		double** TC = ExpandMatrixToSize(C, 0, 0, size);

		t_st = omp_get_wtime();
		Strassen(TA, TB, TC, size, linearMultBlockSize);
		t_ed = omp_get_wtime() - t_st;
		for (int i = 0; i < sizeA; i++)
			for (int j = 0; j < sizeC; j++)
			{
				C[i][j] = TC[i][j];
			}
		for (int i = 0; i < size; i++)
		{
			delete[] TA[i];
			delete[] TB[i];
			delete[] TC[i];
		}
		delete[] TA;
		delete[] TB;
		delete[] TC;

		return t_ed;
	}
}

double Shtrassen_MultiplicationParallelGuided(double** A, double** B, double** C, int sizeA, int sizeB, int sizeC)
{
	int linearMultBlockSize = 64;
	double t_st = 0, t_ed = -1;
	int size = sizeA;
	double time = 0;

	if (sizeA == sizeB && sizeA == sizeC && Validate_size(size, linearMultBlockSize))
	{
		t_st = omp_get_wtime();
		StrassenParallelGuided(A, B, C, size, linearMultBlockSize);
		time = omp_get_wtime() - t_st;
		return time;
	}
	else
	{

		if (size < sizeB) size = sizeB;
		if (size < sizeC) size = sizeC;
		if (size != Find_Valid_Size(size, linearMultBlockSize))
		{
			size = Find_Valid_Size(size, linearMultBlockSize);
		}
#pragma align(4)
		double** TA = ExpandMatrixToSize(A, sizeA, sizeB, size);
#pragma align(4)
		double** TB = ExpandMatrixToSize(B, sizeB, sizeC, size);
#pragma align(4)
		double** TC = ExpandMatrixToSize(C, 0, 0, size);

		t_st = omp_get_wtime();
		StrassenParallelGuided(TA, TB, TC, size, linearMultBlockSize);
		t_ed = omp_get_wtime() - t_st;

		for (int i = 0; i < sizeA; i++)
			for (int j = 0; j < sizeC; j++)
			{
				C[i][j] = TC[i][j];
			}
		for (int i = 0; i < size; i++)
		{
			delete[] TA[i];
			delete[] TB[i];
			delete[] TC[i];
		}
		delete[] TA;
		delete[] TB;
		delete[] TC;

		return t_ed;
	}
}

double Shtrassen_MultiplicationParallelSections(double** A, double** B, double** C, int sizeA, int sizeB, int sizeC)
{
	int linearMultBlockSize = 64;
	double t_st = 0, t_ed = -1;
	int size = sizeA;
	double time = 0;

	if (sizeA == sizeB && sizeA == sizeC && Validate_size(size, linearMultBlockSize))
	{
		t_st = omp_get_wtime();
		StrassenParallelSections(A, B, C, size, linearMultBlockSize);
		time = omp_get_wtime() - t_st;
		return time;
	}
	else
	{

		if (size < sizeB) size = sizeB;
		if (size < sizeC) size = sizeC;
		if (size != Find_Valid_Size(size, linearMultBlockSize))
		{
			size = Find_Valid_Size(size, linearMultBlockSize);
		}
#pragma align(4)
		double** TA = ExpandMatrixToSize(A, sizeA, sizeB, size);
#pragma align(4)
		double** TB = ExpandMatrixToSize(B, sizeB, sizeC, size);
#pragma align(4)
		double** TC = ExpandMatrixToSize(C, 0, 0, size);

		t_st = omp_get_wtime();
		StrassenParallelSections(TA, TB, TC, size, linearMultBlockSize);
		t_ed = omp_get_wtime() - t_st;

		for (int i = 0; i < sizeA; i++)
			for (int j = 0; j < sizeC; j++)
			{
				C[i][j] = TC[i][j];
			}
		for (int i = 0; i < size; i++)
		{
			delete[] TA[i];
			delete[] TB[i];
			delete[] TC[i];
		}
		delete[] TA;
		delete[] TB;
		delete[] TC;

		return t_ed;
	}
}

double TestShtrassen_Multiplication(double**& matrix1, double**& matrix2, double**& matrix3, int& sizeA, int& sizeB, int& sizeC)
{
	return Shtrassen_Multiplication(matrix1, matrix2, matrix3, sizeA, sizeB, sizeC);
}
double TestShtrassen_MultiplicationParallelGuided(double**& matrix1, double**& matrix2, double**& matrix3, int& sizeA, int& sizeB, int& sizeC)
{
	return Shtrassen_MultiplicationParallelGuided(matrix1, matrix2, matrix3, sizeA, sizeB, sizeC);
}
double TestShtrassen_MultiplicationParallelSections(double**& matrix1, double**& matrix2, double**& matrix3, int& sizeA, int& sizeB, int& sizeC)
{
	return Shtrassen_MultiplicationParallelSections(matrix1, matrix2, matrix3, sizeA, sizeB, sizeC);
}

double AvgTrustedInterval(double& avg, vector<double>& times, int& cnt)
{
	double sd = 0, newAVg = 0;
	int newCnt = 0;
	for (int i = 0; i < cnt; i++)
	{
		sd += (times[i] - avg) * (times[i] - avg);
	}
	sd /= (cnt - 1.0);
	sd = sqrt(sd);
	for (int i = 0; i < cnt; i++)
	{
		if (avg - sd <= times[i] && times[i] <= avg + sd)
		{
			newAVg += times[i];
			newCnt++;
		}
	}
	if (newCnt == 0) newCnt = 1;
	return newAVg / newCnt;
}

double TestIter(void* Funct, double**& a, double**& b, double**& c, int& A, int& B, int& C)
{
	double curtime = 0, avgTime = 0, avgTimeT = 0, correctAVG = 0;
	int iterations = 2;
	c = new double* [A];
	c[0] = new double[A * C];
	for (int i = 1; i < A; i++)
		c[i] = &c[0][i * C];

	fillMatrixZero(c, A, C);
	vector<double> Times(iterations);
	for (int i = 0; i < iterations; i++)
	{
		// Запуск функции и получение времени в миллисекундах
		curtime = ((*(TestFunctTempl)Funct)(a, b, c, A, B, C)) * 1000;
		Times[i] = curtime;
		avgTime += curtime;
		cout << "+";
	}

	cout << endl;
	// Вычисление среднеарифметического по всем итерациям и вывод значения на экран
	avgTime /= iterations;
	cout << "AvgTime:" << avgTime << endl;
	// Определения среднеарифметического значения в доверительном интервале по всем итерациям и вывод значения на экран
	avgTimeT = AvgTrustedInterval(avgTime, Times, iterations);
	cout << "AvgTimeTrusted:" << avgTimeT << endl;

	delete[] c[0];
	delete[] c;
	return avgTimeT;
}

void test_functions(void** Functions, vector<string> fNames)
{
	int nd = 0;
	double** a, ** b, ** c;
	double times[4][11][3];
	int A = 1000, B = 1000, C = 1000;
	for (A = 1000; A <= 2500; A += 500)
	{
		a = new double* [A];
		b = new double* [B];
		a[0] = new double[A * B];
		b[0] = new double[B * C];
		for (int i = 1; i < A; i++)
			a[i] = &a[0][i * B];
		for (int i = 1; i < B; i++)
			b[i] = &b[0][i * C];

		for (int threads = 1; threads <= 4; threads++)
		{
			omp_set_num_threads(threads);
			//перебор алгоритмов по условиям
			for (int alg = 0; alg <= 10; alg++)
			{
				if (threads == 1)
				{
					if (alg == 0 || alg == 4 || alg == 8) {
						times[nd][alg][0] = TestIter(Functions[alg], a, b, c, A, B, C);
						times[nd][alg][1] = times[nd][alg][0];
						times[nd][alg][2] = times[nd][alg][0];
					}
				}
				else
				{
					if (alg != 0 && alg != 4 && alg != 8)
					{
						times[nd][alg][threads - 2] = TestIter(Functions[alg], a, b, c, A, B, C);
					}
				}
			}
		}
		delete[] a[0];
		delete[] a;
		delete[] b[0];
		delete[] b;
		nd++;
		B += 300;
		C += 500;
	}
	ofstream fout("output.txt");
	fout.imbue(locale("Russian"));
	for (int ND = 0; ND < 4; ND++)
	{
		switch (ND)
		{
		case 0:
			cout << "\n----------A = 1000*1000 B = 1000*1000 C = 1000*1000----------" << endl;
			break;
		case 1:
			cout << "\n----------A = 1500*1300 B = 1300*1500 C = 1500*1500----------" << endl;
			break;
		case 2:
			cout << "\n----------A = 2000*1600 B = 1600*2000 C = 2000*2000----------" << endl;
			break;
		case 3:
			cout << "\n----------A = 2500*1900 B = 1900*2500 C = 2500*2500----------" << endl;
			break;
		default:
			break;
		}


		for (int alg = 0; alg <= 10; alg++)
		{
			for (int threads = 1; threads <= 4; threads++)
			{
				if (threads == 1)
				{
					if (alg == 0 || alg == 4 || alg == 8) {
						cout << "Поток " << threads << " --------------" << endl;
						cout << fNames[alg] << "\t" << times[ND][alg][0] << " ms." << endl;
						fout << times[ND][alg][0] << endl;
					}
				}
				else
				{
					if (alg != 0 && alg != 4 && alg != 8)
					{
						cout << "Поток " << threads << " --------------" << endl;
						cout << fNames[alg] << "\t" << times[ND][alg][threads - 2] << " ms." << endl;
						fout << times[ND][alg][threads - 2] << endl;
					}
				}
			}
		}
	}
	fout.close();
}

int main() {

	setlocale(LC_ALL, "RUS");

	void** Functions = new void* [11] { TestfillMatrix, TestfillMatrixParallelStatic, TestfillMatrixParallelDynamic, TestfillMatrixParallelGuided, TestmultiplyMatrixV4,
		TestmultiplyMatrixV4ParallelStatic, TestmultiplyMatrixV4ParallelDynamic, TestmultiplyMatrixV4ParallelGuided,
		TestShtrassen_Multiplication, TestShtrassen_MultiplicationParallelGuided, TestShtrassen_MultiplicationParallelSections };
	vector<string> function_names = { "Заполнение матрицы", "Параллельное заполнение матрицы for(static)", "Параллельное заполнение матрицы for(dynamic)",
		"Параллельное заполнение матрицы for(guided)", "Перемножение матриц", "Параллельная перемножение FOR(static)",
		"Параллельная перемножение FOR(dynamic)", "Параллельная перемножение FOR(guided)", "Перемножение методом Штрассена",
		"Перемножение методом Штрассена for(guided)",  "Перемножение методом Штрассена (Sections)" };

	test_functions(Functions, function_names);

	return 0;
}
