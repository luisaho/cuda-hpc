/*****************************************************
 * CG Solver (HPC Software Lab)
 *
 * Parallel Programming Models for Applications in the 
 * Area of High-Performance Computation
 *====================================================
 * IT Center (ITC)
 * RWTH Aachen University, Germany
 * Author: Tim Cramer (cramer@itc.rwth-aachen.de)
 * 	   Fabian Schneider (f.schneider@itc.rwth-aachen.de)
 * Date: 2010 - 2015
 *****************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENACC
# include <openacc.h>
#endif

#ifdef CUDA
# include <cuda.h>
#endif

#include "solver.h"
#include "output.h"


/* ab <- a' * b */
void vectorDot(const floatType* a, const floatType* b, const int n, floatType* ab){
	int i;
	floatType temp;
	temp=0;
	for(i=0; i<n; i++){
		temp += a[i]*b[i];
	}
	*ab = temp;
}

/* y <- ax + y */
__device__ void axpy(const floatType a, const floatType* x, const int n, floatType* y){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n){
		y[i]=a*x[i]+y[i];
	}
}

/* y <- x + ay */
__device__ void xpay(const floatType* x, const floatType a, const int n, floatType* y){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n){
		y[i]=x[i]+a*y[i];
	}
}

/* y <- A*x
 * Remember that A is stored in the ELLPACK-R format (data, indices, length, n, nnz, maxNNZ). */
__device__ void matvec(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* x, floatType* y){
	int j, k;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		floatType temp = 0.0;
		for (j = 0; j < length[i]; j++) {
			k = j * n + i;
			temp += data[k] * x[indices[k]];
		}
	y[i] = temp;
	}
}

/* nrm <- ||x||_2 */
void nrm2(const floatType* x, const int n, floatType* nrm){
	int i;
	floatType temp;
	temp = 0;
	for(i = 0; i<n; i++){
		temp+=(x[i]*x[i]);
	}
	*nrm=sqrt(temp);
}

//kernels
__global__ void matvec_kernel(const int d_n, const int d_nnz, const int d_maxNNZ, const floatType* d_data, const int* d_indices, const int* d_length, const floatType* d_x, floatType* d_y){

matvec(d_n, d_nnz, d_maxNNZ, d_data, d_indices, d_length, d_x, d_y);

}

__global__ void axpy_kernel(const floatType a, const floatType* d_x, const int d_n, floatType* d_y){

axpy(a, d_x, d_n, d_y);

}

__global__ void xpay_kernel(const floatType* d_x, const floatType a, const int d_n, floatType* d_y){

xpay(d_x, a, d_n, d_y);

}

/***************************************
 *         Conjugate Gradient          *
 *   This function will do the CG      *
 *  algorithm without preconditioning. *
 *    For optimiziation you must not   *
 *        change the algorithm.        *
 ***************************************
 r(0)    = b - Ax(0)
 p(0)    = r(0)
 rho(0)    =  <r(0),r(0)>                
 ***************************************
 for k=0,1,2,...,n-1
   q(k)      = A * p(k)                 
   dot_pq    = <p(k),q(k)>             
   alpha     = rho(k) / dot_pq
   x(k+1)    = x(k) + alpha*p(k)      
   r(k+1)    = r(k) - alpha*q(k)     
   check convergence ||r(k+1)||_2 < eps  
	 rho(k+1)  = <r(k+1), r(k+1)>         
   beta      = rho(k+1) / rho(k)
   p(k+1)    = r(k+1) + beta*p(k)      
***************************************/
void cg(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* b, floatType* x, struct SolverConfig* sc){

	dim3 threadsPerBlock(128);
	dim3 blocksPerGrid(n/threadsPerBlock.x);

	floatType* r, *p, *q;
	floatType *d_p, *d_q;
	floatType alpha, beta, rho, rho_old, dot_pq, bnrm2;
	int iter;
 	double timeMatvec_s;
 	double timeMatvec=0;
	
	/* allocate memory */
	r = (floatType*)malloc(n * sizeof(floatType));
	p = (floatType*)malloc(n * sizeof(floatType));
	q = (floatType*)malloc(n * sizeof(floatType));
	cudaMalloc(&d_p, n * sizeof(floatType));
	cudaMalloc(&d_q, n * sizeof(floatType));

	DBGMAT("Start matrix A = ", n, nnz, maxNNZ, data, indices, length)
	DBGVEC("b = ", b, n);
	DBGVEC("x = ", x, n);

	/* r(0)    = b - Ax(0) */


	int *d_n, *d_nnz, *d_maxNNZ, *d_indices, *d_length;
	floatType *d_data, *d_x, *d_r, *d_b;
	cudaMalloc(&d_n, sizeof(int));
	cudaMalloc(&d_nnz, sizeof(int));
	cudaMalloc(&d_maxNNZ, sizeof(int));
	cudaMalloc(&d_indices, n * maxNNZ * sizeof(int));
	cudaMalloc(&d_length, n * sizeof(int));
	cudaMalloc(&d_data, n * maxNNZ * sizeof(floatType));
	cudaMalloc(&d_x, n * sizeof(floatType));
	cudaMalloc(&d_r, n * sizeof(floatType));
	cudaMalloc(&d_b, n * sizeof(floatType));

	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_maxNNZ, &maxNNZ, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, indices, n * maxNNZ * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_length, length, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_data, data, n * maxNNZ * sizeof(floatType), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, n * sizeof(floatType), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, r, n * sizeof(floatType), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_b, b, n * sizeof(floatType), cudaMemcpyHostToDevice);	

	timeMatvec_s = getWTime();
	matvec_kernel<<<blocksPerGrid,threadsPerBlock>>>(*d_n, *d_nnz, *d_maxNNZ, d_data, d_indices, d_length, d_x, d_r);
	timeMatvec += getWTime() - timeMatvec_s;
	xpay_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_b, -1.0, *d_n, d_r);
	DBGVEC("r = b - Ax = ", r, n);
	

	/* Calculate initial residuum */
	nrm2(r, n, &bnrm2);
	bnrm2 = 1.0 /bnrm2;

	/* p(0)    = r(0) */
	memcpy(p, r, n*sizeof(floatType));
	DBGVEC("p = r = ", p, n);

	/* rho(0)    =  <r(0),r(0)> */
	vectorDot(r, r, n, &rho);
	printf("rho_0=%e\n", rho);

	for(iter = 0; iter < sc->maxIter; iter++){
		DBGMSG("=============== Iteration %d ======================\n", iter);
		/* q(k)      = A * p(k) */
		timeMatvec_s = getWTime();
		matvec_kernel<<<blocksPerGrid,threadsPerBlock>>>(*d_n, *d_nnz, *d_maxNNZ, d_data, d_indices, d_length, d_x, d_r);
		timeMatvec += getWTime() - timeMatvec_s;
		DBGVEC("q = A * p= ", q, n);

		/* dot_pq    = <p(k),q(k)> */
		vectorDot(p, q, n, &dot_pq);
		DBGSCA("dot_pq = <p, q> = ", dot_pq);

		/* alpha     = rho(k) / dot_pq */
		alpha = rho / dot_pq;
		DBGSCA("alpha = rho / dot_pq = ", alpha);

		/* x(k+1)    = x(k) + alpha*p(k) */
		axpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(alpha, d_p, *d_n, d_x);
		DBGVEC("x = x + alpha * p= ", x, n);

		/* r(k+1)    = r(k) - alpha*q(k) */
		axpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(-alpha, d_q, *d_n, d_r);
		DBGVEC("r = r - alpha * q= ", r, n);


		rho_old = rho;
		DBGSCA("rho_old = rho = ", rho_old);


		/* rho(k+1)  = <r(k+1), r(k+1)> */
		vectorDot(r, r, n, &rho);
		DBGSCA("rho = <r, r> = ", rho);

		/* Normalize the residual with initial one */
		sc->residual= sqrt(rho) * bnrm2;


   	
		/* Check convergence ||r(k+1)||_2 < eps
		 * If the residual is smaller than the CG
		 * tolerance specified in the CG_TOLERANCE
		 * environment variable our solution vector
		 * is good enough and we can stop the 
		 * algorithm. */
		printf("res_%d=%e\n", iter+1, sc->residual);
		if(sc->residual <= sc->tolerance)
			break;


		/* beta      = rho(k+1) / rho(k) */
		beta = rho / rho_old;
		DBGSCA("beta = rho / rho_old= ", beta);

		/* p(k+1)    = r(k+1) + beta*p(k) */
		xpay_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_r, beta, *d_n, d_p);
		DBGVEC("p = r + beta * p> = ", p, n);

	}

	/* Store the number of iterations and the 
	 * time for the sparse matrix vector
	 * product which is the most expensive 
	 * function in the whole CG algorithm. */

	//x update
	cudaMemcpy(x, d_x, n * sizeof(floatType), cudaMemcpyDeviceToHost);

	sc->iter = iter;
	sc->timeMatvec = timeMatvec;

	/* Clean up */
	cudaFree(d_n);	
	cudaFree(d_nnz);	
	cudaFree(d_maxNNZ);	
	cudaFree(d_data);	
	cudaFree(d_length);	
	cudaFree(d_x);	
	cudaFree(d_r);	
	cudaFree(d_indices);
	cudaFree(d_q);	
	cudaFree(d_p);	
	cudaFree(d_b);	

	free(r);
	free(p);
	free(q);
	

}


