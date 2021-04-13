#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#define LARGENUMBER 1e20
#define N 1000000
#define SMALL 10000  // arbitrary
#define n 5
#define NREP 10  // arbitrary

#define min(a,b) ((a)<(b))?(a):(b)


void initMat(float *a, int dim)
{  
	 int i, j;

	 for (i=0; i<dim; i++) {
		 for (j=0; j<dim; j++) {
			a[i*dim+j] = (10.f*rand())/((float)RAND_MAX);
		 }
		}
}

void productMat_seq(float *a, float *b, float *c, int dim) 
{  			
   float sum;
	int row, col, k;
	for (row=0; row<dim; row++){
   	for (col=0; col<dim; col++){
		   sum = 0.f;
		   for (k=0; k<dim; k++){
		       sum += a[row*dim+k]*b[k*dim+col];
		   }
		c[row*dim+col] = sum;
		//printf("%f",c);
	  }
	}

}

void productMat_omp_optimized(float *a, float *b, float *c, int dim, int nthr) 
{
	int part_rows, th_id;
  part_rows = dim/nthr;
  
  omp_set_num_threads(nthr); //set the number of threads
  #pragma omp parallel default(none) shared(a,b,c,part_rows) 	private(dim)
  {
   float sum;
	int row, col, k;
	
	#pragma omp for schedule(guided,part_rows)
	for (row=0; row<dim; row++){
   	for (col=0; col<dim; col++){
		   sum = 0.f;
		   for (k=0; k<dim; k++){
		       sum += a[row*dim+k]*b[k*dim+col];
		   }
			c[row*dim+col] = sum;
		  }
		}
	}
}

int main(int argc, char *argv[argc])
{

  //float m;
  double timing,t0;
  int numthr;
  int dim;
  
  int l;
  
  
  if(argc>1)
    numthr=atoi(argv[1]);  
  else {
    //directive in case a user doesn't know the parameters
    printf("\nProgram parameters are: [number of threads] [matrix dimension].");
    printf("\nNo number of threads provided. Assuming 4.");
    numthr=4;
  }

  //the second argument is the number of rows of the first matrix. if none is provided, use 100.
  if(argc>2)
    dim=atoi(argv[2]);
  else {
    printf("\nNo matrix dimension provided. Using 100-by-100 square matrices.");
    dim=100;
  }
  
  float *a = calloc(dim*dim, sizeof(float));
  if(a==NULL) printf("\nUnable to allocate memory for matrix a.\n");
  float *b = calloc(dim*dim, sizeof(float));
  if(b==NULL) printf("\nUnable to allocate memory for matrix b.\n");
  float *c = calloc(dim*dim, sizeof(float));
  if(c==NULL) printf("\nUnable to allocate memory for matrix c.\n");
  
    //get the two matrices to be multiplied
  printf("\nGenerating two random matrices...");
//  srand( (unsigned int)time(NULL) );
  srand(time(NULL)); // Initialize the random number generator
  printf("a (%d x %d)...",dim,dim);
  initMat(a, dim);
  printf("Done. b (%d x %d)...",dim,dim);
  initMat(b, dim);
  printf("Done.\n");
  

  printf("Multiplying the matrices...");
  timing = 0.0;
  

  if (numthr == 1){
  for (l=0; l<NREP; l++) {
    t0 = omp_get_wtime();
    productMat_seq(a,b,c,dim);
    timing = timing +  omp_get_wtime() - t0;
    }
   }
  else{
  for (l=0; l<NREP; l++) {
    t0 = omp_get_wtime();
    productMat_omp_optimized(a,b,c,dim,numthr);
    timing = timing +  omp_get_wtime() - t0;
    }
  }
  printf("Done.\n");


  
 /* printf("Matrice a:\n");
  for(int row=0; row<dim; row++){
  		for(int col=0; col<dim; col++) printf("%1.4f ",a[row*dim+col]);
		printf("\n");
  }
  
  printf("Matrice b:\n");
  for(int row=0; row<dim; row++){
  		for(int col=0; col<dim; col++) printf("%1.4f ",b[row*dim+col]);
		printf("\n");
		
  }
  
  printf("Matrice résultante de la multiplication des 2 matrices a et b:\n");
  for(int row=0; row<dim; row++){
  		for(int col=0; col<dim; col++) printf("%1.4f ",c[row*dim+col]);
		printf("\n");	
		
	}
	*/
	
  printf("It took the program %lf s to complete a x b = c using %d thread(s) with dimension %d \n", timing/(1.0*NREP),numthr,dim);


  //free all of the memory
  free(a);
  free(b);
  free(c);
  return 0;
}
