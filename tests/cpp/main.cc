/* b.c */
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  
  int rank, size;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  while (1){
    printf ("Hello from process %d of %d\n", rank, size);
    sleep(10);
  }
  
  MPI_Finalize();
  return 0;
}