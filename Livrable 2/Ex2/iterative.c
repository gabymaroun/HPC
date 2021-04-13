#include "iterative.h"
#include "vector.h"
#include "matrix.h"
#include "sparsematrix.h"

#define MAXITER 1000000

int solveJacobi(matrix A, vector b, vector x, double tol){
  /* Solves Ax=b with Jacobi iterative method whithin the giben tolerance (tol)
   * Vector x contains x_0 as the initial guess
   * Returns the iteration number.
   */
  int k=0;
  vector r = createVector(A.n);
  // r initial
  prodMV(A, x, r);
  axpy(-1.0, r, b, r);
  while(k<MAXITER && norme2(r)>tol) {
    // x = x + D^-1r
    for (int i=0; i < A.n; i++)
      x.data[i]  = x.data[i] + r.data[i]/A.data[i*A.n+i];
    // r = b - Ax
    prodMV(A, x, r);
    axpy(-1.0, r, b, r);
    k += 1;
  }
  destroyVector(r);
  return k;
}

int solveGaussSeidel(matrix A, vector b, vector x, double tol){
  /* Solves Ax=b with Gauss-Seidel iterative method
   * Vector x contains x_0 as the initial guess
  * Returns the iteration number.
  */
  int k=0;
  vector r = createVector(A.n);
  // r initial
  prodMV(A, x, r);
  axpy(-1.0, r, b, r);
  while(k<MAXITER && norme2(r)>tol) {
    for (int i=0; i < A.n; i++) {
      // Compute new x
      x.data[i] = b.data[i];
      for (int j=0; j < A.n; j++)
        if(i!=j)
          x.data[i] -= A.data[i*A.n+j]*x.data[j];
      x.data[i] /= A.data[i*A.n+i];
    }
    // r = b - Ax
    prodMV(A, x, r);
    axpy(-1.0, r, b, r);
    k += 1;
  }
  destroyVector(r);
  return k;
}

int solveSparseGaussSeidel(sMatrix A, vector b, vector x, double tol) {
  /* Solves Ax=b with Gauss-Seidel iterative method. A is a sparse matrix
   * Vector x contains x_0 as the initial guess
  * Returns the iteration number.
  */
  int k=0;
  vector r = createVector(A.n);
  vector AdiagInv = createVector(A.n);
  // A diagonal computation
  for (int i = 0; i < A.n; i++) {
    for (int nz = 0; nz < A.nnzi[i+1]-A.nnzi[i]; nz++) {
      int j = A.j[A.nnzi[i]+nz];
      if(i==j) AdiagInv.data[i] = 1.0/A.data[A.nnzi[i]+nz];
    }
  }
  // r initial
      #pragma omp parallel default(none) shared(A, x, r, b, k, tol,AdiagInv)
  prodSMV(A, x, r);
  axpy(-1.0, r, b, r);
  while(k<MAXITER && norme2(r)>tol) {
  #pragma omp for schedule(static)
    for (int i=0; i < A.n; i++) {
      // Compute new x
      x.data[i] = b.data[i];
      for (int nz = 0; nz < A.nnzi[i+1]-A.nnzi[i]; nz++) {
        int j = A.j[A.nnzi[i]+nz];
        if(i!=j)
          x.data[i] -= A.data[A.nnzi[i]+nz]*x.data[j];
      }
      x.data[i] *= AdiagInv.data[i];
    }
    // r = b - Ax
    prodSMV(A, x, r);
    axpy(-1.0, r, b, r);
    k += 1;
  }
  destroyVector(r);
  destroyVector(AdiagInv);
  return k;
}


int solveSparseJacobi(sMatrix A, vector b, vector x, double tol) {
  /* Solves Ax=b with Jacobi iterative method
   * Vector x contains x_0 as the initial guess
  * Returns the iteration number.
  */
  int k=0;
  vector r = createVector(A.n);
  vector AdiagInv = createVector(A.n);
  vector rnew = createVector(A.n);
  for (int i=0; i < A.n; i++) {
    x.data[i] = 0.0;
    r.data[i] = b.data[i];
  }
  // A diagonal computation
  for (int i = 0; i < A.n; i++) {
    for (int nz = 0; nz < A.nnzi[i+1]-A.nnzi[i]; nz++) {
      int j = A.j[A.nnzi[i]+nz];
      if(i==j) AdiagInv.data[i] = 1.0/A.data[A.nnzi[i]+nz];
    }
  }
    #pragma omp parallel default(none) shared(x,r,AdiagInv,b,rnew, k, tol) 	private(A)
  while(k<MAXITER && norme2(r)>tol) {
    // x = x + D^-1r
    #pragma omp for schedule(static)
    for (int i=0; i < A.n; i++)
      x.data[i]  = x.data[i] + r.data[i]*AdiagInv.data[i];
    // r = b - Ax
    prodSMV(A, x, rnew);
    #pragma omp for schedule(static)
    for (int i=0; i < A.n; i++)
      r.data[i] = b.data[i] - rnew.data[i];
    k += 1;
  }
  destroyVector(AdiagInv);
  destroyVector(r);
  destroyVector(rnew);
  return k;
}
