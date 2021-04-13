#include<stdio.h>
#include<math.h>
#include<time.h>

#include "vector.h"
#include "matrix.h"
#include "sparsematrix.h"
#include "direct.h"
#include "iterative.h"
#include "mesh2dp1.h"
#include "elem2dp1.h"
#include "integration.h"

const integForm2d iF = GAUSS3PT2D;
const integForm iF1d = GAUSSLEGENDRE3PT1D;

// Problem data
const double T=10.0;
const double lambda = 1e-2;
const double uext = 1.0;
int isDirichlet(int id) {return id==5;}
double dirichlet(double x, double y, double t, int id) {return 17.0+3.0*cos(2*M_PI*t/T);}
int isNeumann(int id) {return id==1||id==2||id==3;}
double neumann(double x, double y, int id) {return 0.0;}
int isFourier(int id) {return id==4;}
double fourier(double x, double y, double t, int id) {return lambda*(uext-3*sin(2*M_PI*t/T));}
double conductivite(elem2dp1 e) {
  if (e.elemId == 6) return 1e-1; /* Air */
  if (e.elemId == 7) return 1e-4; /* Cloisons */
  if (e.elemId == 8) return 1e-4; /* plafond */
  return NAN;
}

void initSMMS(mesh2dp1 mesh, sMatrix *M, sMatrix *S) {
  printf("Assemble matrices with %d compute points using %s integration formula on %d elements.\n",
         mesh.nbNodes, iF.name, mesh.nbElems);
  *S = initSMatrixStructureFromMesh2dp1(mesh);
  *M = initSMatrixStructureFromMesh2dp1(mesh);

  // Compute matrix
  int n = mesh.dof;
  vector gradphihati = createVector(2);
  vector gradphihatj = createVector(2);
  vector gradphii = createVector(2);
  vector gradphij = createVector(2);

  for(int iK=0; iK<mesh.nbElems; iK++){
    elem2dp1 e = mesh.elems[iK];
    for(int k=0; k<iF.n; k++) {
      double xhat = iF.ptsX[k];
      double yhat = iF.ptsY[k];
      double w = iF.weights[k]*fabs(e.detJ);
      for (int ii = 0; ii < n; ii++) {
        int i = e.index[ii];
        gradphihati.data[0] = e.dxphihat[ii](xhat, yhat);
        gradphihati.data[1] = e.dyphihat[ii](xhat, yhat);
        prodMV(e.Jinv, gradphihati, gradphii);
        for (int jj = 0; jj < n; jj++) {
          int j = e.index[jj];
          int sj = meshTools2dp1.idxOfNbor(mesh,i,j);
          gradphihatj.data[0] = e.dxphihat[jj](xhat, yhat);
          gradphihatj.data[1] = e.dyphihat[jj](xhat, yhat);
          prodMV(e.Jinv, gradphihatj, gradphij);
          // Mass matrix
          M->data[M->nnzi[i]+sj] += w * e.phihat[ii](xhat, yhat) * e.phihat[jj](xhat, yhat);
          // Stiffness matrix
          S->data[S->nnzi[i]+sj] += w * conductivite(e) * (gradphii.data[0]*gradphij.data[0] + gradphii.data[1]*gradphij.data[1]);
        }
      }
    }
  }
  destroyVector(gradphihati);
  destroyVector(gradphihatj);
  destroyVector(gradphii);
  destroyVector(gradphij);
}

void imposeBCS(mesh2dp1 mesh, sMatrix *S) {
/* Impose boundary conditions except for Dirichlet  */
  for (int iK = 0; iK < mesh.nbSide; iK++) {
    elem2dp1_edge e = mesh.side[iK];
    if(isFourier(e.edgeId)) {
      for(int k=0; k<iF1d.n; k++) {
        double that = iF1d.pts[k];
        double x,y;
        meshTools2dp1.xFromTHat_edge(mesh, e, that, &x, &y);
        double w = iF1d.weights[k]*fabs(e.detJ);
        for (int ii = 0; ii < e.dof; ii++) {
          int i = e.index[ii];
          for (int jj = 0; jj < e.dof; jj++) {
            int j = e.index[jj];
            int sj = meshTools2dp1.idxOfNbor(mesh,i,j);
            S->data[S->nnzi[i]+sj] += lambda * w * e.phihat[ii](that) * e.phihat[jj](that);
          }
        }
      }
    }
  }
}

void imposeBCrhs(mesh2dp1 mesh, vector *b, double t) {
/* Impose boundary conditions except for Dirichlet  */
  for (int iK = 0; iK < mesh.nbSide; iK++) {
    elem2dp1_edge e = mesh.side[iK];
    if(isNeumann(e.edgeId)) {
      for(int k=0; k<iF1d.n; k++) {
        double that = iF1d.pts[k];
        double x,y;
        meshTools2dp1.xFromTHat_edge(mesh, e, that, &x, &y);
        double w = iF1d.weights[k]*fabs(e.detJ);
        double fk = neumann(x, y, e.edgeId);
        for (int ii = 0; ii < e.dof; ii++) {
          int i = e.index[ii];
          b->data[i] += w*fk*e.phihat[ii](that);
        }
      }
    }
    if(isFourier(e.edgeId)) {
      for(int k=0; k<iF1d.n; k++) {
        double that = iF1d.pts[k];
        double x,y;
        meshTools2dp1.xFromTHat_edge(mesh, e, that, &x, &y);
        double w = iF1d.weights[k]*fabs(e.detJ);
        double fr = fourier(x, y, t, e.edgeId);
        for (int ii = 0; ii < e.dof; ii++) {
          int i = e.index[ii];
          b->data[i] += w*fr*e.phihat[ii](that);
        }
      }
    }
  }
}
void imposeDirichletSystem(mesh2dp1 mesh, sMatrix *A, vector *b, double t) {
  for (int iK = 0; iK < mesh.nbSide; iK++) {
    elem2dp1_edge e = mesh.side[iK];
    if(isDirichlet(e.edgeId)) {
      int i = e.index[0];
      for(int j=A->nnzi[i]; j<A->nnzi[i+1]; j++) {
        A->data[j] = 0.0;
      }
      A->data[A->nnzi[i] + meshTools2dp1.idxOfNbor(mesh,i,i)] = 1.0;
      b->data[i] = dirichlet(mesh.coordsX.data[i], mesh.coordsY.data[i], t, e.edgeId);
    }
  }
}

int main(int argc, char *argv[argc])
{
  printf("TP1\n");
  mesh2dp1 mesh = meshTools2dp1.createMeshFromFile("meshtp10_0.05-0.01.msh");
  vector u = createVector(mesh.nbNodes);
  vector buold = createVector(mesh.nbNodes);
  vector uold = createVector(mesh.nbNodes);
  vector b = createVector(mesh.nbNodes);
  vector f = createVector(mesh.nbNodes);
  vector fold = createVector(mesh.nbNodes);
  sMatrix S, M, A, B;
  double dt, t;
  int k, outputfreq = 0;

  initSMMS(mesh, &M, &S);
  imposeBCS(mesh, &S);
  imposeBCrhs(mesh, &fold, 0.0);
  resetVector(uold, 17.0);


  A = createSMatrixFromStruct(S);
  B = createSMatrixFromStruct(S);

  dt =0.1;
  for (int i = 0; i < M.nnzi[mesh.nbNodes]; i++) {
    A.data[i] = M.data[i] + 0.5*dt*S.data[i];
    B.data[i] = M.data[i] - 0.5*dt*S.data[i];
  }

  if(outputfreq > 0) openHDF5MeshAndTimedData(mesh, "tp1");
  for(k=1; k<=T/dt; k++) {
    t=k*dt;
    prodSMV(B, uold, buold);
    imposeBCrhs(mesh, &f, t);
    for (int i = 0; i < M.n; i++)
      b.data[i] = buold.data[i] + 0.5*dt*(f.data[i]+f.data[i]);
    imposeDirichletSystem(mesh, &A, &b, t);

    solveSparseGaussSeidel(A, b, u ,norme2(b)*1e-11);
    for (int i = 0; i < M.n; i++) {
      uold.data[i] = u.data[i];
      fold.data[i] = f.data[i];
    }
    if(outputfreq > 0 && k%outputfreq==0) stepHDF5TimedData(mesh, u, k, t, "tp1");
  }
  if(outputfreq > 0 && (k-1)%outputfreq>0) stepHDF5TimedData(mesh, u, k-1, t, "tp1");
  if(outputfreq > 0) closeHDF5MeshAndTimedData("tp1");


  destroyMesh2dp1(mesh);
  destroyVector(u);
  destroyVector(uold);
  destroyVector(b);
  destroySMatrix(S);
  destroySMatrix(M);
  destroySMatrix(A);
  destroySMatrix(B);
  return 0;
}
