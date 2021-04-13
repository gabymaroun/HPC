#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <math.h>


void rand_mat(float *a, int dim)
{  
	 int i, j;

	 for (i=0; i<dim; i++) {
		 for (j=0; j<dim; j++) {
			a[i*dim+j] = (10.f*rand())/((float)RAND_MAX);
		 }
		}
}


//A=Id
void Id_mat(float *a, int n)
{
  int i,j; 
 
  for(i=0; i<n; i++) 
    for(j=0; j<n; j++) {
      if(i==j){
	a[i*n+j] = 1;
	}
      else{
	a[i*n+j] = 0;
	}	
	}	
}

//B=(jn+i)
void B_mat(float *a, int n){
  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      a[i*n+j]=j*n+i;
    }

  }
}


void multip_mat(float *c,float *a,float *b, int n, int nloc, int row){

  int irow;
  for (int i=0; i<nloc; i++) {
    irow=row+i;
    for (int j=0; j<nloc; j++) {
      c[irow*nloc+j]=0;
      for (int k=0; k<n; k++) {
	
	c[irow*nloc+j]+=a[i*n+k]*b[k*nloc+j];
      }
    }
  }
  
}

void print_mat(float *c, int dim)
{
  int i,j;
  for(i=0; i<dim; i++)
  {
    for(j=0; j<dim; j++)
      printf("%1.1f ", c[j+i*dim]);
    printf("\n");
  }
  
}

void check(float *b, float *c, int dim)
{
  int i,j, isok=0;
  for(i=0; i<dim; i++)
  {
    for(j=0; j<dim; j++){
    
      if(c[j+i*dim]!=b[j+i*dim]){
      	isok=1;
      	break;
      }
    }
  }
  if(isok==1){
	  	printf("Not OK\n");
    }
  else{
  		printf("OK\n");
  }
}

int main(int argc, char *argv[]) {
  int rank , size , l;
  char proc_name [50];

  MPI_Init(&argc, &argv); // Initialisation de l'env. MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Rang dans le comm.
  MPI_Comm_size(MPI_COMM_WORLD , &size); // Taille du comm.

 
  MPI_Status status;

  int n=1120;
  
  if(argc > 1){
   n=atoi(argv[1]);
   }
   
  //int nloc=n/(float)size+1;
  int nloc=n/(float)size;
  //printf("%d\n",nloc);
  float* A;
  float* B;
  float* C;
  float* tmp;
  float* A_loc;
  float* B_loc;
  float* C_loc;
   
  // Allocation of local matrices
  A_loc = calloc(nloc*n, sizeof(float));
  B_loc = calloc(n*nloc, sizeof(float));
  C_loc = calloc(n*nloc, sizeof(float));


  MPI_Datatype rows;
   
  MPI_Type_contiguous(n*nloc, MPI_FLOAT, &rows );
  MPI_Type_commit(&rows);
      

  double startTime,endTime;
  int Nrep=10;
    
  if (rank == 0) {
    
     
    // Allocations of full matrices
    A = calloc(n*n, sizeof(float));
    B = calloc(n*n, sizeof(float));
    
    // Initialize A and B matrices
    Id_mat(A,n);
    B_mat(B,n);
    /*rand_mat(A,n);
	 rand_mat(B,n);*/

  }
  C = calloc(n*n, sizeof(float));
  tmp = calloc(n*n, sizeof(float));
  
  startTime=MPI_Wtime();
  for (int irep=0; irep<Nrep; irep++){
    //if (rank==0){
    
      MPI_Datatype ligne;
      MPI_Type_vector(nloc, n, n, MPI_FLOAT, &ligne );
      MPI_Type_commit(&ligne);

      MPI_Datatype cols;      
      MPI_Type_vector(n, nloc, n, MPI_FLOAT, &cols );
      MPI_Type_commit(&cols);

      //B to rank 0
      /*MPI_Sendrecv(B, 1,
      		   cols, rank+1, rank+1,
      		   B_loc, n*nloc,
      		   MPI_FLOAT, rank-1, rank-1,
      		   MPI_COMM_WORLD, &status);*/
   
     
      
      
     //}
     //B to rank != 0
     int index,index0;
      for (int irank=0; irank<size; irank++){
			MPI_Send(B+nloc*irank, 1, cols,
		 	irank, irank, MPI_COMM_WORLD);
      }
      //A from rank 0/ A to others 
      for (int t=0; t<size; t++){
			index0=((t)*nloc)%n;
			//printf("%d",index0);
			MPI_Sendrecv(A+n*index0, 1,
					  rows, 0, size+t,
					  A_loc,1,
					  rows, 0, size+t,
					  MPI_COMM_WORLD, &status);
					  
			 //Compute C_loc
			multip_mat(C_loc,A_loc,B_loc,n,nloc,index0);
			for (int irank=0; irank<size; irank++){
					  index=((irank+t)*nloc)%n;
					  //printf("%d\n",index);
					  MPI_Send(A+n*index, 1, rows,
						   irank, size+irank*size+t, MPI_COMM_WORLD);
			
					}
			
		}      

    
    
      // B as columns / A to rank 0
    //if (rank!=0){
		   MPI_Recv(B_loc, nloc*n, MPI_FLOAT,
		 	       rank-1, rank, MPI_COMM_WORLD, &status);
			//printf("%d\n",rank);
		   for (int t=0; t<size; t++){

				MPI_Recv(A_loc, 1,rows,
					 rank-1, size+rank*size+t, MPI_COMM_WORLD, &status);

				 //matrix multiplication
			 
				multip_mat(C_loc,A_loc,B_loc,n,nloc,((rank+t)*nloc)%n);
				//printf("%d",((rank+t)*nloc-rank)%n);
      	}
		//}
 		
    // Gathering C_loc data into C matrix
    
    MPI_Allgather(C_loc, n*nloc,
	       MPI_FLOAT, tmp, n*nloc,
	       MPI_FLOAT, MPI_COMM_WORLD);


    //reorganisation of matrix C, small time compare to the all time after doing a test
    //if (rank==0){
      

      int i=0,j=0,i1=0,j1=0;
      for (int irank=0; irank<size; irank++){
      	for (int iloc=0; iloc<nloc; iloc++){
      	  for (int jrank=0; jrank<size ; jrank++){
      	    for (int jloc=0; jloc<nloc; jloc++){
      	      i=jrank+size*iloc;
      	      j=jloc+nloc*irank;
      	      i1=irank*nloc+iloc;
      	      j1=jrank*nloc+jloc;
      	      C[i*n+j]=tmp[i1*n+j1];      
      	    }
      	  }
      	}	
      }

    //}
  }
	//printf("%d\n",tmp[20]);
 endTime=MPI_Wtime();
 
 
  //print time into file
  //if (rank==0){
  
      FILE* file;
      file= fopen("scalability.dat","a+");
      fprintf(file,"%d %g \n",size, (endTime-startTime));
      
      //option to print matrices by using p or check if B=C by using c
      if(argc>2){
			
		   if(strcmp(argv[2],"p")==0 || strcmp(argv[2],"print")==0){
				//print_mat(C,n);
				 printf("A=[\n");
				 print_mat(A,n);
				 printf("B=[\n");
				 print_mat(B,n);
				 printf("C=[\n");
				 print_mat(C,n);
				}
			else if(strcmp(argv[2],"c")==0 || strcmp(argv[2],"check")==0){
			
				check(B,C,n);
			}
		}
   //}
  if(rank == 0) {
    free(A);free(B);
  }
  free(C);free(tmp);
  free(A_loc);free(B_loc);free(C_loc);
  MPI_Finalize();

    

  return 0;
}
