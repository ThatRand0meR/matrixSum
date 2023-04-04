#include <cuda_profiler_api.h>
#include <helper_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>

// funkcja wypisująca zawartość przekazanej tablicy w kształcie kwadratu
void printSquare(int arrayWidth, float* array) {
	for (int i = 0; i < (arrayWidth); i++) {
		for (int j = 0; j < (arrayWidth); j++) {
			std::cout << array[i * (arrayWidth)+j] << " ";
		}
		std::cout << std::endl;
	}
}


// funkcja sumująca elementy tablicy oddalone o r - do kodu sekwencyjnego
float sumNeighbours(int currentOutElementY, int currentOutElementX, int n, int r, float* neighboursArray, int widthOfOutArray) {
	int currentElement = n * currentOutElementY + currentOutElementX;
	float sum = 0;
	for (int i = -r; i < r + 1; i++) {
		for (int j = -r; j < r + 1; j++) {
			sum += neighboursArray[currentElement % n + (r + i) + (currentElement / n + r + j) * n];
		}
	}
	return sum;
}

// funkcja obliczająca sekwencyjnie - do kodu sekwencyjnego
void calculateSequential(int n, int r, float* arrayIn, float* arrayOut, int widthOfOutArray) {
	for (int i = 0; i < widthOfOutArray; i++) {
		for (int j = 0; j < widthOfOutArray; j++) {
			arrayOut[i * widthOfOutArray + j] = sumNeighbours(i, j, n, r, arrayIn, widthOfOutArray);
		}
	}
}

double generateRandomFloat(float min_value, float max_value)
{
	std::default_random_engine eng;
	unsigned long int t = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	eng.seed(t);
	static std::mt19937 gen(eng());
	std::uniform_real_distribution<> dist(min_value, max_value);
	return dist(gen);
}



// funkcja obliczająca na GPU
__global__ void kernel(int n, int r, int k, float* inArray, float* outArray, int widthOfOutArray)
{
	// id X i Y w gridzie pierwszego threada w danym bloku, 
	// mnożone razy k - dany blok oblicza BS*k x BS elementów tablicy
	int blockStartingIdX = blockIdx.x * blockDim.x * k;
	int blockStartingIdY = blockIdx.y * blockDim.y;

	// id X i Y w gridzie danego threada w danym bloku, 
	// mnożone razy k - dany thread oblicza k elementów tablicy jadąc w poziomie
	int threadIdX = blockStartingIdX + (threadIdx.x * k);
	int threadIdY = blockStartingIdY + threadIdx.y;

	// thready ktore są poza zakresem na ktorym chcemy pracowac nie powinny probowac dostac sie do elementow tablicy, moze to powodowac 
	// probe dostepu do niezadeklarowanego obszaru pamieci 
	if ((threadIdX < widthOfOutArray) && (threadIdY < widthOfOutArray)) {
		float sum = 0;
		// iterowanie przez k - ilosc elementow tablicy ktore ma obliczyc dany watek
		for (int currentK = 0; currentK < k; currentK++) {
			// ponowne sprawdzenie czy nie odwolujemy sie do wartosci poza zakresem
			if ((threadIdX + currentK < widthOfOutArray) && (threadIdY < widthOfOutArray)) {
				// iterowanie po sasiadach danego elementu tablicy
				for (int i = -r; i < r + 1; i++) {
					for (int j = -r; j < r + 1; j++) {
						sum += inArray[threadIdX + currentK + (r + i) + (threadIdY + r + j) * n];
					}
				}
				outArray[threadIdY * widthOfOutArray + threadIdX + currentK] = sum;
				// reset sumy dla k>1
				sum = 0;
			}
		}
	}
}


// kernel shared - oblicznie na gpu z przechowywaniem w pamieci wspoldzielonej
__global__ void kernel_shared(int n, int r, int k, float* inArray, float* outArray, int widthOfOutArray,
	int elementsToMoveTosharedPerThread)
{
	// id X i Y w gridzie pierwszego threada w danym bloku, 
	int blockStartingIdX = blockIdx.x * blockDim.x;
	int blockStartingIdY = blockIdx.y * blockDim.y;

	// id X i Y w gridzie danego threada w danym bloku, 
	int threadIdX = blockStartingIdX + threadIdx.x ;
	int threadIdY = blockStartingIdY + threadIdx.y;

	// [debug] numer aktualnego bloku
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	//if (blockId > 3) return;
	// [debug] wylaczenie wysztkich poza pierwszym blokiem
	//	if (blockIdx.x != 0 || blockIdx.y != 0) {
	//	return ;
	//	}
	//int threadId = (blockId * blockDim.x * blockDim.y + threadIdx.y * blockDim.y + threadIdx.x) * k;

	// zatrzymanie watkow, dla n-2r niepodzielnego przez blockDim*k, przed proba obliczania wartosci ktorymi nie powinny sie zajmowac
	int thisBlockWidhtNeeded = blockDim.x * k;
	int thisBlockHeightNeeded = blockDim.y;
	if (blockIdx.x == gridDim.x - 1 && widthOfOutArray % (blockDim.x * k)!=0) {
		thisBlockWidhtNeeded = widthOfOutArray % (blockDim.x * k);
	}

	extern __shared__ float shared[];

	int threadNrInBlock = threadIdx.y * blockDim.x + threadIdx.x;
	// index od ktorego watek powinien zaczac swoja czesc przenoszenia danych do shared
	int firstIndexToMoveToShared = threadNrInBlock * elementsToMoveTosharedPerThread;

	// sprawdzenie czy nie odwolujemy sie do wartosci poza zakresem
	if (firstIndexToMoveToShared < (blockDim.x + r + r) * (blockDim.x + r + r) * k) {
		for (int iteration = 0; iteration < elementsToMoveTosharedPerThread * k; iteration++) {
			int currentIndexToMoveToShared = firstIndexToMoveToShared * k + iteration;
			// ponowne sprawdzenie czy nie odwolujemy sie do wartosci poza zakresem
			if (currentIndexToMoveToShared < (blockDim.x + r + r) * (blockDim.y + r + r) * k) {
				shared[currentIndexToMoveToShared] = inArray[currentIndexToMoveToShared % (blockDim.x * k + r + r) +
					(blockStartingIdY + currentIndexToMoveToShared / (blockDim.x * k + r + r)) * n + blockStartingIdX * k];
			}
		}
	}

	// synchronizacja watkow po wrzuceniu do pamieci shared
	__syncthreads();

	// obliczanie wyniku z użyciem shared

	// sprawdzenie czy nie odwolujemy sie do wartosci poza zakresem
	if ((threadIdX < widthOfOutArray) && (threadIdY < widthOfOutArray)) {
		float sum = 0;

		for (int currentK = 0; currentK < k; currentK++) {
			// ponowne sprawdzenie czy nie odwolujemy sie do wartosci poza zakresem
			if ((threadIdx.x * k + currentK < thisBlockWidhtNeeded) && (threadIdY< widthOfOutArray)) {
				for (int i = -r; i < r + 1; i++) {
					for (int j = -r; j < r + 1; j++) {
						//sum += shared[0];
						sum += shared[threadIdx.x * k  + (r + i) + currentK + (threadIdx.y + r + j) * (blockDim.x * k + r + r)];
					}
				}
				//outArray[threadIdY * widthOfOutArray + threadIdX * k + currentK] = threadIdx.x * k + currentK +
				//  (threadIdx.y) * (thisBlockWidhtNeeded * k + r + r);
			//	outArray[threadIdY * widthOfOutArray + threadIdX * k + currentK] = shared[threadIdx.x * k +currentK +
				//(threadIdx.y) * (thisBlockWidhtNeeded * k + r + r)];
				outArray[threadIdY * widthOfOutArray + threadIdX *k + currentK] = sum;
			//	outArray[threadIdY * widthOfOutArray + threadIdX * k + currentK] = blockId;
				sum = 0;
			}
		}
	}
}


int main()
{
	const int n = 256; //szerokosc tablicy |============|
	const int r = 12; //zasieg aka promien |===========| przetestować dla 2 różnych wartości, większej i mniejszej niż BS
	const int blockWidth = 8; // szerokosc bloku |====| 8, 16 lub 32
	const int k = 1; //ilosc obliczen na watek |====|
	const int sizeOfInArray = n * n;
	const int widthOfOutArray = (n - r * 2);
	const int sizeOfOutArray = widthOfOutArray * widthOfOutArray;


	if (r * 2 >= n) {
		std::cout << "zbyt duże R dla podanego N" << std::endl;
		return 0;
	}

	float* arrayIn;

	//======================Setup Memory for zero copy 
	// 
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	if (!prop.canMapHostMemory)
		exit(0);
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&arrayIn, sizeOfInArray * sizeof(float), cudaHostAllocMapped);
	//===========


	float* arrayOut = new float[sizeOfOutArray];

	//===========
	//wypelnienie talbicy wejsciowej danymi
	//
	for (int i = 0; i < sizeOfInArray; i++) {
		//arrayIn[i] = 1;
		arrayIn[i] = i;
		//arrayIn[i] = generateRandomFloat(0, 20);
	}
	//===========


	
	//wypisanie elementów tablicy wejsciowej
//	printSquare(n, arrayIn);
	
	
	//===========
	//uruchomienie funkcji obliczajacej sekwencyjnie
	//calculateSequential(n, r, arrayIn, arrayOut, widthOfOutArray);	TODO

	//===========



	//===========
	//wypisanie elementów obliczonych sekwencyjnie
	//std::cout << "=========sekwencyjnie============" << std::endl;
//	printSquare(widthOfOutArray, arrayOut);
	//===========


	//================start CUDA
	std::cout << prop.name << std::endl;

	//wybranie karty graficznej
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!");
		return 1;
	}

	//deklaracja pointera do tablicy wejsciowej dla GPU
	float* arrayIn_onDevice;

	//inicjalizacja pointera do tablicy wejsciowej dla GPU
	cudaStatus = cudaHostGetDevicePointer(&arrayIn_onDevice, arrayIn, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaHostGetDevicePointer failed!");
		return 1;
	}


	//deklaracja pointera do tablicy wyjsciowej dla hosta oraz GPU
	float* arrayOutCUDA_onHost;
	float* arrayOutCUDA_onDevice;


	//inicjalizacja pointera do tablicy wyjsciowej dla hosta
	cudaStatus = cudaHostAlloc((void**)&arrayOutCUDA_onHost, sizeof(float) * sizeOfOutArray, cudaHostAllocDefault);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaHostAlloc failed!");
		return 1;
	}

	//inicjalizacja pointera do tablicy wyjsciowej dla GPU
	cudaStatus = cudaHostGetDevicePointer((void**)&arrayOutCUDA_onDevice, (void*)arrayOutCUDA_onHost, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaHostGetDevicePointer failed!");
		return 1;
	}


	// ilosc obliczen do wykonania na blok
	float calculationsPerBlock = ceil((float)blockWidth * (float)blockWidth );
	// wymagana ilosc blokow
	float blockCount = ceil((float)sizeOfOutArray / calculationsPerBlock);
	// szerokosc i wysokosc grida
	float blockRowCount = ceil(sqrt(blockCount));


	std::cout << "calculationsPerBlock :" << calculationsPerBlock << std::endl;
	std::cout << "blockCount :" << blockCount << std::endl;
	std::cout << "blockRowCount :" << blockRowCount << std::endl;

	// szerokosc grida i bloku dla kernela
	dim3 threads(blockWidth, blockWidth);
	dim3 blocks(ceil((float)blockRowCount / k), blockRowCount);

	std::cout << "ceil((float)blockRowCount / k) :" << ceil((float)blockRowCount / k) << std::endl;

	// wielkosc shared dla kazdego bloku
	int elementsToMoveToshared = (blockWidth + r + r) * (blockWidth + r + r);
	// ilosc przeniesien do shared ktore musi wykonac kazdy watek
	int elementsToMoveTosharedPerThread = ceil((float)elementsToMoveToshared / (float)(blockWidth * blockWidth));
	std::cout << "elementsToMoveTosharedPerThread :" << elementsToMoveTosharedPerThread << std::endl;



	// ====================================
	// funkcje do mierzenia parametrow 
	// 
	// Allocate CUDA events that we'll use for timing

	cudaStream_t stream;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	// Create and start timer
	checkCudaErrors(cudaStreamSynchronize(stream));

	// Record the start event
	checkCudaErrors(cudaEventRecord(start, stream));
	// ===================
	// ilosc uruchomien kernela - dla usrednienia wynikow
	int nIter = 20;
	// ===================


	// wywolanie kernela nIter razy
	for (int j = 0; j < nIter; j++) {
		kernel << <blocks, threads >> > (n, r, k, arrayIn_onDevice, arrayOutCUDA_onDevice, widthOfOutArray);
		//kernel_shared << <blocks, threads, (elementsToMoveToshared * sizeof(float) * k) >> > (n, r, k, arrayIn_onDevice, arrayOutCUDA_onDevice, widthOfOutArray, elementsToMoveTosharedPerThread);
	}

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, stream));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	// Compute and print the performance
	float msecPerIteration = msecTotal / nIter;
	double flopsPerIteration = (r + 1) * (r+1) * static_cast<double>(n - r - r) *
		static_cast<double>(n - r - r);
	double gigaFlops =
		(flopsPerIteration * 1.0e-9f) / (msecPerIteration / 1000.0f);	
	printf(
			"Total TIME = %.3f msec\n",
		msecTotal);

	printf(
		"Performance= %.2f GFlop/s, Time= %.3f msec, OpsAmount= %.0f Ops,"
		" WorkgroupSize= %u threads/block\n",
		gigaFlops, msecPerIteration, flopsPerIteration, threads.x * threads.y);
	 
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	//==========================================
	// koniec pracy na GPU
	//==========================================
	cudaDeviceSynchronize();
	//==========================================
	//printSquare(widthOfOutArray, arrayOutCUDA_onDevice);
	// wypisanie wyniku z GPU
	//==========================================

	//==========================================
	// test poprawnosci - porownanie gpu z sekwencyjnym
	/*	TODO
	bool noMistakes = true;
	for (int i = 0; i < sizeOfOutArray; i++)
	{
		if (arrayOut[i] != arrayOutCUDA_onHost[i]) {
			noMistakes = false;
		//	std::cout << "mistake on i:" << i << " expected: " << arrayOut[i] << " got: " << arrayOutCUDA_onHost[i] << std::endl;
		}

	}
	if (noMistakes)
	{
		std::cout << "No mistakes" << std::endl;

	}
	*/
	//==========================================



	//==========================================
	// zwolnienie pamieci
	cudaFreeHost(arrayIn);
	cudaFreeHost(arrayOutCUDA_onHost);
	delete[] arrayOut;
	//==========================================

	return 0;
}
