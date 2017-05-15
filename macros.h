
#define THREADS_PER_BLOCK 1024
#define BLOCKSIZE (1024)
#define UPDIV(N, threadsPerBlock) (((N) + (threadsPerBlock) - 1) / (threadsPerBlock))

//cancer dataset
#define N (513)
#define P (9)
#define SQRT_P (3)
#define NUM_BLOCKS (3)

//loan dataset
/*#define N (655158)
#define P (13)
#define SQRT_P (4)
#define NUM_BLOCKS (4)


//marketing dataset
#define N (33909)
#define P (6)
#define SQRT_P (3)
#define NUM_BLOCKS (4)*/
