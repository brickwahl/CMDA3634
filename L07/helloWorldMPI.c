#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"


int main(int argc, char** argv) {
  
  //every MPI program must start with an initalize
  //always do this first
  MPI_Init(&argc, &argv);
  
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); //this tells MPI to get the rank of this process globally and to store the result in rank
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  if (rank == 0) {
    int N = 10;
    int destRank = 1;
    int tag = 1;

    MPI_Send(&N, 1, MPI_INT, destRank, tag, MPI_COMM_WORLD); //pointer to data were sending, numbeer of entries to send, data type of each entry, rank of destintion, tags the message with an identifier, f									lag to full MPI network
  }

  else if (rank == 1) {
    
    int N;
    int sourceRank =0;
    int tag = 1;
    MPI_Status status;

    MPI_Recv(&N, 1, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, &status);

    printf("Rank %d recieved a message from rank %d: value = %d\n", rank, sourceRank, N);
  } 

  // all MPI programs must end with a finalize
  MPI_Finalize();
  return 0;
}
