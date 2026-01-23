#include "nussinov_lib.h"
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Standard Nussinov Algorithm Implementation

bool is_valid_pair_nussinov(char a, char b) {
  return (a == 'A' && b == 'U') || (a == 'U' && b == 'A') ||
         (a == 'C' && b == 'G') || (a == 'G' && b == 'C') ||
         (a == 'G' && b == 'U') || (a == 'U' && b == 'G');
}

static void traceback_internal(int n, int (*dp)[n], const char *sequence, int i,
                               int j, Pair *pairs, int *count) {
  if (i >= j)
    return;

  if (dp[i][j] == dp[i + 1][j]) {
    traceback_internal(n, dp, sequence, i + 1, j, pairs, count);
  } else if (dp[i][j] == dp[i][j - 1]) {
    traceback_internal(n, dp, sequence, i, j - 1, pairs, count);
  } else if (is_valid_pair_nussinov(sequence[i], sequence[j]) && (j - i > 3) &&
             dp[i][j] == 1 + dp[i + 1][j - 1]) {
    pairs[*count].i = i;
    pairs[*count].j = j;
    (*count)++;
    traceback_internal(n, dp, sequence, i + 1, j - 1, pairs, count);
  } else {
    for (int k = i + 1; k < j; k++) {
      if (dp[i][j] == dp[i][k] + dp[k + 1][j]) {
        traceback_internal(n, dp, sequence, i, k, pairs, count);
        traceback_internal(n, dp, sequence, k + 1, j, pairs, count);
        return;
      }
    }
  }
}

Pair *nussinov_predict(const char *seq, int *out_count) {
  int n = strlen(seq);

  // Dynamic Programming Table
  int (*dp)[n] = malloc(sizeof(int[n][n]));
  if (!dp) {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

// Initialize DP
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      dp[i][j] = 0;
    }
  }

  // Fill DP Table
  for (int k = 1; k < n; k++) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n - k; i++) {
      int j = i + k;

      int max_score = dp[i + 1][j];
      if (dp[i][j - 1] > max_score)
        max_score = dp[i][j - 1];

      if (is_valid_pair_nussinov(seq[i], seq[j]) && (j - i > 3)) {
        int pair_score = 1 + dp[i + 1][j - 1];
        if (pair_score > max_score)
          max_score = pair_score;
      }

      for (int split = i + 1; split < j; split++) {
        int split_score = dp[i][split] + dp[split + 1][j];
        if (split_score > max_score)
          max_score = split_score;
      }

      dp[i][j] = max_score;
    }
  }

  // Traceback
  int max_pairs = dp[0][n - 1];
  *out_count = 0;

  Pair *pairs = malloc(max_pairs * sizeof(Pair));
  if (max_pairs > 0) {
    traceback_internal(n, dp, seq, 0, n - 1, pairs, out_count);
  }

  free(dp);
  return pairs;
}

#ifdef NUSSINOV_STANDALONE
int main(int argc, char *argv[]) {
  char *sequence;
  if (argc > 1) {
    sequence = argv[1];
  } else {
    sequence = "CUUACCAUCGGGUUAGAGGAG";
  }

  int n = strlen(sequence);
  printf("SEQ : %s\n", sequence);
  printf("Len : %d\n", n);

  int count = 0;
  Pair *pairs = nussinov_predict(sequence, &count);

  printf("Max Pairs: %d\n", count);

  // Print Dot-Bracket Structure
  char *structure = malloc(n + 1);
  memset(structure, '.', n);
  structure[n] = '\0';

  for (int i = 0; i < count; i++) {
    structure[pairs[i].i] = '(';
    structure[pairs[i].j] = ')';
  }

  printf("Structure: %s\n", structure);

  // CSV Output
  FILE *fpt = fopen("output.csv", "w+");
  fprintf(fpt, "dot-bracket, pairs\n");
  fprintf(fpt, "%s, %d\n", structure, count);
  fclose(fpt);

  free(pairs);
  free(structure);

  return 0;
}
#endif