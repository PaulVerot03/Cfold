#ifndef NUSSINOV_LIB_H
#define NUSSINOV_LIB_H

typedef struct {
  int i;
  int j;
} Pair;

/*
 * Predicts secondary structure using Nussinov algorithm.
 * Returns an array of Pairs (caller handles memory, though this implementation
 * might just return a malloced array). count: pointer to int to store the
 * number of pairs found.
 */
Pair *nussinov_predict(const char *seq, int *count);

#endif
