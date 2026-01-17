#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <omp.h>

#define BACKBONE_DIST 6.0f   /* Ideal P-to-P distance (Ångströms) - RNA backbone spacing */
#define BASE_PAIR_DIST 10.0f /* Ideal Watson-Crick pair distance (Ångströms) */
#define REPULSION_DIST 12.0f /* Excluded volume cutoff (Ångströms) - atoms can't come closer */
#define K_BOND 1.0f          /* Spring constant for bonds (kcal/mol/Å²) */
#define K_REPEL 1.0f         /* Spring constant for repulsion (kcal/mol/Å²) */
#define LEARNING_RATE 0.2f   /* Initial step size for molecular dynamics (Ångströms) */
#define ITERATIONS 50000     /* Number of MD iterations - allows time for convergence */
#define MAX_FORCE 10.0f      /* Force clamping limit - prevents numerical instability (Å/iter) */

#define MAX_SEQ_LEN 8096 /* Maximum sequence length supported */

typedef struct
{
    float x, y, z;
} Vec3;

Vec3 vec_add(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 vec_sub(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }

Vec3 vec_scale(Vec3 a, float s) { return (Vec3){a.x * s, a.y * s, a.z * s}; }

float vec_dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3 vec_cross(Vec3 a, Vec3 b) { return (Vec3){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }

float vec_norm(Vec3 a) { return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z); }

Vec3 vec_normalize(Vec3 a)
{
    float n = vec_norm(a);
    if (n < 1e-6f)
        return (Vec3){1.0f, 0.0f, 0.0f}; // Default fallback
    return vec_scale(a, 1.0f / n);
}

Vec3 vec_clamp(Vec3 v, float max_len)
{
    float n = vec_norm(v);
    if (isnan(n) || isinf(n))
        return (Vec3){0, 0, 0};
    if (n > max_len)
    {
        float scale = max_len / n;
        return (Vec3){v.x * scale, v.y * scale, v.z * scale};
    }
    return v;
}

typedef struct
{
    Vec3 col[3]; /* Three column vectors forming rotation axes */
} Mat3;

Vec3 mat_mul_vec(Mat3 m, Vec3 v)
{
    return (Vec3){
        m.col[0].x * v.x + m.col[1].x * v.y + m.col[2].x * v.z,
        m.col[0].y * v.x + m.col[1].y * v.y + m.col[2].y * v.z,
        m.col[0].z * v.x + m.col[1].z * v.y + m.col[2].z * v.z};
}

typedef struct
{
    int i, j;
} Pair;

typedef struct
{
    char name[5];     /* Atom name (e.g., "P", "C4'", "N9") */
    char res_name[4]; /* Residue name (usually 3-letter code) */
    char chain;       /* Chain identifier */
    int res_seq;      /* Residue sequence number */
    Vec3 coords;      /* 3D coordinates in Ångströms */
} Atom;

/*
 * Returns: true if pair is chemically valid, false otherwise
 */
bool is_valid_bp(char a, char b)
{
    if ((a == 'A' && b == 'U') || (a == 'U' && b == 'A'))
        return true;
    if ((a == 'G' && b == 'C') || (a == 'C' && b == 'G'))
        return true;
    if ((a == 'G' && b == 'U') || (a == 'U' && b == 'G'))
        return true;
    return false;
}

int memo[MAX_SEQ_LEN][MAX_SEQ_LEN];

/*
 * Time: O(n³), Space: O(n²)
 */
void nussinov_compute(const char *seq, int n, bool *mask)
{
    memset(memo, 0, sizeof(memo));
    for (int k = 1; k < n; k++)
    {
        for (int i = 0; i < n - k; i++)
        {
            int j = i + k;
            int score = memo[i + 1][j];
            if (score < memo[i][j - 1])
                score = memo[i][j - 1];

            if (!mask[i] && !mask[j])
            {
                if (is_valid_bp(seq[i], seq[j]) && (j - i > 3))
                {
                    int p_score = 1 + memo[i + 1][j - 1];
                    if (p_score > score)
                        score = p_score;
                }
            }

            for (int split = i + 1; split < j; split++)
            {
                int s_score = memo[i][split] + memo[split + 1][j];
                if (s_score > score)
                    score = s_score;
            }
            memo[i][j] = score;
        }
    }
}

/*
 * Time: O(n²) in worst case, typically much faster
 */
void traceback(const char *seq, int i, int j, Pair *pairs, int *pair_count, bool *mask)
{
    if (i >= j)
        return;

    if (memo[i][j] == memo[i + 1][j])
    {
        traceback(seq, i + 1, j, pairs, pair_count, mask);
    }
    else if (memo[i][j] == memo[i][j - 1])
    {
        traceback(seq, i, j - 1, pairs, pair_count, mask);
    }
    else if (!mask[i] && !mask[j] && is_valid_bp(seq[i], seq[j]) && (j - i > 3) && memo[i][j] == 1 + memo[i + 1][j - 1])
    {
        pairs[*pair_count].i = i;
        pairs[*pair_count].j = j;
        (*pair_count)++;
        traceback(seq, i + 1, j - 1, pairs, pair_count, mask);
    }
    else
    {
        for (int k = i + 1; k < j; k++)
        {
            if (memo[i][j] == memo[i][k] + memo[k + 1][j])
            {
                traceback(seq, i, k, pairs, pair_count, mask);
                traceback(seq, k + 1, j, pairs, pair_count, mask);
                return;
            }
        }
    }
}

/*
 * Time: O(n³) for each pass (dominated by DP split loop)
 */
Pair *predict_structure(const char *seq, int *out_count)
{
    int n = strlen(seq);
    bool *mask = calloc(n, sizeof(bool));
    Pair *all_pairs = malloc(n * sizeof(Pair)); /* Max pairs is n/2 */
    int total_pairs = 0;

    nussinov_compute(seq, n, mask);

    Pair *new_pairs = malloc(n * sizeof(Pair));
    int new_count = 0;
    traceback(seq, 0, n - 1, new_pairs, &new_count, mask);

    /* Mark paired bases so they won't pair again */
    for (int k = 0; k < new_count; k++)
    {
        all_pairs[total_pairs++] = new_pairs[k];
        mask[new_pairs[k].i] = true;
        mask[new_pairs[k].j] = true;
    }

    if (new_count > 0)
    {
        new_count = 0;
        nussinov_compute(seq, n, mask);
        traceback(seq, 0, n - 1, new_pairs, &new_count, mask);

        for (int k = 0; k < new_count; k++)
        {
            if (!mask[new_pairs[k].i] && !mask[new_pairs[k].j])
            {
                all_pairs[total_pairs++] = new_pairs[k];
                mask[new_pairs[k].i] = true;
                mask[new_pairs[k].j] = true;
            }
        }
    }

    free(mask);
    free(new_pairs);
    *out_count = total_pairs;
    return all_pairs;
}

#define K_ANGLE 0.2f
#define ANGLE_TARGET 2.0f

void add_force(Vec3 *forces, int idx, Vec3 f)
{
    forces[idx].x += f.x;
    forces[idx].y += f.y;
    forces[idx].z += f.z;
}

void apply_angular_force(Vec3 p_prev, Vec3 p_curr, Vec3 p_next, Vec3 *forces, int i_prev, int i_curr, int i_next)
{
    Vec3 v1 = vec_sub(p_prev, p_curr);
    Vec3 v2 = vec_sub(p_next, p_curr);
    float n1 = vec_norm(v1);
    float n2 = vec_norm(v2);

    if (n1 < 1e-6f || n2 < 1e-6f)
        return; /* Avoid division by near-zero */

    float dot = vec_dot(v1, v2);
    float cos_theta = dot / (n1 * n2);

    if (cos_theta > 1.0f)
        cos_theta = 1.0f;
    if (cos_theta < -1.0f)
        cos_theta = -1.0f;

    float theta = acosf(cos_theta);

    float target_chord = 2.0f * BACKBONE_DIST * sinf(ANGLE_TARGET / 2.0f);

    Vec3 chord_vec = vec_sub(p_next, p_prev);
    float chord_dist = vec_norm(chord_vec);

    float chord_mag = K_ANGLE * (chord_dist - target_chord);
    Vec3 f_chord = vec_scale(vec_normalize(chord_vec), chord_mag);

    add_force(forces, i_prev, f_chord);
    add_force(forces, i_next, vec_scale(f_chord, -1.0f));
}

/*
 *   Outer loop: 50,000 iterations
 *   Backbone bonds: O(n)
 *   Base pairs: O(#pairs)
 *   Repulsion: O(n²) per iteration (pairwise comparison)
 *   Total: O(n² * ITERATIONS) = O(50000 * n²)
 *
 * TYPICAL RUNTIMES:
 *   n = 100 nt: ~1-2 seconds
 *   n = 500 nt: ~50-100 seconds
 *   n = 1000 nt: 200-400 seconds
 */
void physics_fold_serial(int n, Vec3 *coords, Pair *pairs, int pair_count)
{
    printf("running in serial mode for small N...\n");
    Vec3 *forces = malloc(n * sizeof(Vec3));

    float current_learn_rate = LEARNING_RATE;

    for (int iter = 0; iter < ITERATIONS; iter++)
    {
        memset(forces, 0, n * sizeof(Vec3));
        current_learn_rate = LEARNING_RATE * expf(-5.0f * (float)iter / ITERATIONS);

        for (int i = 0; i < n - 1; i++)
        {
            int j = i + 1;
            Vec3 d = vec_sub(coords[j], coords[i]);
            float dist = vec_norm(d);
            float mag = K_BOND * (dist - BACKBONE_DIST);
            Vec3 f = vec_scale(vec_normalize(d), mag);
            add_force(forces, i, f);
            add_force(forces, j, vec_scale(f, -1.0f));

            if (i > 0)
            {
                apply_angular_force(coords[i - 1], coords[i], coords[j], forces, i - 1, i, j);
            }
        }

        for (int p = 0; p < pair_count; p++)
        {
            int i = pairs[p].i;
            int j = pairs[p].j;
            Vec3 d = vec_sub(coords[j], coords[i]);
            float dist = vec_norm(d);
            float mag = 5.0f * K_BOND * (dist - BASE_PAIR_DIST);
            Vec3 f = vec_scale(vec_normalize(d), mag);
            add_force(forces, i, f);
            add_force(forces, j, vec_scale(f, -1.0f));
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 2; j < n; j++)
            {
                Vec3 d = vec_sub(coords[j], coords[i]);
                float dist = vec_norm(d);
                if (dist < REPULSION_DIST && dist > 1e-6f)
                {
                    float mag = -K_REPEL * (REPULSION_DIST - dist);
                    Vec3 f = vec_scale(vec_normalize(d), mag);
                    add_force(forces, i, f);
                    add_force(forces, j, vec_scale(f, -1.0f));
                }
            }
        }

        for (int i = 0; i < n; i++)
        {
            Vec3 f_clamped = vec_clamp(forces[i], MAX_FORCE);
            coords[i] = vec_add(coords[i], vec_scale(f_clamped, current_learn_rate));
        }
    }
    printf("Serial Folding complete.\n");
    free(forces);
}

/*Same soup, just in parallel this time*/
/*using Mr.Lucas' class on High Power Computing */
void physics_fold_parallel(int n, Vec3 *coords, Pair *pairs, int pair_count)
{
    int max_threads = omp_get_max_threads();
    printf("running in parallel mode (threads=%d)...\n", max_threads);

    Vec3 *all_local_forces = malloc(max_threads * n * sizeof(Vec3));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Vec3 *local_forces = &all_local_forces[tid * n];

        float current_learn_rate = LEARNING_RATE;

        for (int iter = 0; iter < ITERATIONS; iter++)
        {
            current_learn_rate = LEARNING_RATE * expf(-5.0f * (float)iter / ITERATIONS);

            memset(local_forces, 0, n * sizeof(Vec3));

#pragma omp for schedule(static) nowait
            for (int i = 0; i < n - 1; i++)
            {
                int j = i + 1;
                Vec3 d = vec_sub(coords[j], coords[i]);
                float dist = vec_norm(d);
                float mag = K_BOND * (dist - BACKBONE_DIST);
                Vec3 f = vec_scale(vec_normalize(d), mag);
                add_force(local_forces, i, f);
                add_force(local_forces, j, vec_scale(f, -1.0f));

                if (i > 0)
                {
                    apply_angular_force(coords[i - 1], coords[i], coords[j], local_forces, i - 1, i, j);
                }
            }

#pragma omp for schedule(static) nowait
            for (int p = 0; p < pair_count; p++)
            {
                int i = pairs[p].i;
                int j = pairs[p].j;
                Vec3 d = vec_sub(coords[j], coords[i]);
                float dist = vec_norm(d);
                float mag = 5.0f * K_BOND * (dist - BASE_PAIR_DIST);
                Vec3 f = vec_scale(vec_normalize(d), mag);
                add_force(local_forces, i, f);
                add_force(local_forces, j, vec_scale(f, -1.0f));
            }

#pragma omp for schedule(dynamic, 4)
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 2; j < n; j++)
                {
                    Vec3 d = vec_sub(coords[j], coords[i]);
                    float dist = vec_norm(d);
                    if (dist < REPULSION_DIST && dist > 1e-6f)
                    {
                        float mag = -K_REPEL * (REPULSION_DIST - dist);
                        Vec3 f = vec_scale(vec_normalize(d), mag);
                        add_force(local_forces, i, f);
                        add_force(local_forces, j, vec_scale(f, -1.0f));
                    }
                }
            }

#pragma omp barrier

#pragma omp for schedule(static)
            for (int i = 0; i < n; i++)
            {
                Vec3 total_f = {0, 0, 0};
                for (int t = 0; t < max_threads; t++)
                {
                    total_f = vec_add(total_f, all_local_forces[t * n + i]);
                }
                Vec3 f_clamped = vec_clamp(total_f, MAX_FORCE);
                coords[i] = vec_add(coords[i], vec_scale(f_clamped, current_learn_rate));
            }
        }
    }
    printf("Parallel Folding complete.\n");
    free(all_local_forces);
}

void physics_fold(int n, Vec3 *coords, Pair *pairs, int pair_count)
{
    printf("Stage 2: Folding in 3D (%d iterations)...\n", ITERATIONS);

    if (n < 150)
    {
        physics_fold_serial(n, coords, pairs, pair_count);
    }
    else
    {
        physics_fold_parallel(n, coords, pairs, pair_count);
    }
}

typedef struct
{
    char name[5];
    Vec3 pos;    
} TemplateAtom;

TemplateAtom BACKBONE_TMPL[] = {
    {"P", {0.000, 0.000, 0.000}},    
    {"OP1", {-0.500, 1.400, 0.000}},  
    {"OP2", {-0.500, -0.800, 1.200}}, 
    {"C5'", {2.500, 1.000, 0.500}}, 
    {"O5'", {1.600, 0.000, 0.000}}, 
    {"C4'", {3.500, 1.000, -0.500}},
    {"O4'", {4.000, 2.200, -0.500}},  
    {"C3'", {4.500, -0.200, -0.500}}, 
    {"O3'", {5.500, -0.500, 0.500}},  
    {"C2'", {4.000, -1.200, -1.200}},
    {"C1'", {4.200, 2.200, -1.500}}  
};
int BB_COUNT = 11;

TemplateAtom BASE_A[] = {
    {"N9", {1.3, 0.5, -0.5}},
    {"C8", {1.8, 1.5, -0.5}}, 
    {"N7", {3.0, 1.5, -0.5}},  
    {"C5", {3.4, 0.2, -0.5}},  
    {"C6", {4.8, -0.1, -0.5}},
    {"N6", {5.5, 1.0, -0.5}},  
    {"N1", {5.1, -1.4, -0.5}},
    {"C2", {4.2, -2.0, -0.5}}, 
    {"N3", {3.0, -1.8, -0.5}},
    {"C4", {2.5, -0.5, -0.5}} 
};
int BASE_A_COUNT = 10;

Mat3 get_frenet_frame(Vec3 prev, Vec3 curr, Vec3 next, bool has_prev)
{
    Vec3 t = vec_sub(next, curr);
    if (vec_norm(t) < 1e-6f)
        t = (Vec3){1.0f, 0.0f, 0.0f};
    else
        t = vec_normalize(t);

    Vec3 n;
    if (!has_prev)
    {
        Vec3 guess = {0.0f, 0.0f, 1.0f};
        if (fabs(vec_dot(t, guess)) > 0.9f)
            guess = (Vec3){0.0f, 1.0f, 0.0f};
        n = vec_normalize(vec_cross(t, guess));
    }
    else
    {
        Vec3 v1 = vec_normalize(vec_sub(prev, curr));
        Vec3 bisector = vec_add(v1, t);
        if (vec_norm(bisector) < 1e-6f)
        {
            Vec3 guess = {0.0f, 0.0f, 1.0f};
            n = vec_normalize(vec_cross(t, guess));
        }
        else
        {
            n = vec_sub(bisector, vec_scale(t, vec_dot(bisector, t)));
            n = vec_normalize(n);
        }
    }

    Vec3 b = vec_cross(t, n);

    Mat3 R;
    R.col[0] = t;
    R.col[1] = vec_scale(n, -1.0f);
    R.col[2] = b;
    return R;
}

void save_pdb(const char *filename, int n, const char *seq, Vec3 *coords)
{
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    FILE *f = fopen(filename, "w");
    if (!f)
        return;

    int atom_id = 1;
    fprintf(f, "HEADER    RNA STRUCTURE       %04d%02d%02d_%02d%02d%02d   XXXX              \n", t->tm_year + 1900, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
    fprintf(f, "TITLE     RNA TERTIARY STRUCTURE PREDICTION                    \n");
    fprintf(f, "REMARK    Generated by Cfold                   \n");
    fprintf(f, "REMARK    Sequence length: %d nucleotides                     \n", n);

    for (int i = 0; i < n; i++)
    {
        Vec3 pos = coords[i];
        Vec3 prev = (i > 0) ? coords[i - 1] : (Vec3){0, 0, 0};
        Vec3 next;
        if (i < n - 1)
            next = coords[i + 1];
        else
            next = vec_add(pos, vec_sub(pos, prev));

        Mat3 R = get_frenet_frame(prev, pos, next, i > 0);

        Vec3 c1_pos = {0, 0, 0};

        for (int k = 0; k < BB_COUNT; k++)
        {
            Vec3 loc = BACKBONE_TMPL[k].pos;
            Vec3 glob = vec_add(pos, mat_mul_vec(R, loc));

            if (strcmp(BACKBONE_TMPL[k].name, "C1'") == 0)
                c1_pos = loc;

            fprintf(f, "ATOM  %5d %-4s %3c A%4d    %8.3f%8.3f%8.3f  1.00  0.00           %c\n",
                    atom_id++, BACKBONE_TMPL[k].name, seq[i], i + 1, glob.x, glob.y, glob.z, BACKBONE_TMPL[k].name[0]);
        }

        char res = seq[i];

        for (int k = 0; k < BASE_A_COUNT; k++)
        {
            Vec3 total_loc = vec_add(c1_pos, BASE_A[k].pos);
            Vec3 glob = vec_add(pos, mat_mul_vec(R, total_loc)); 
            fprintf(f, "ATOM  %5d %-4s %3c A%4d    %8.3f%8.3f%8.3f  1.00  0.00           %c\n",
                    atom_id++, BASE_A[k].name, res, i + 1, glob.x, glob.y, glob.z, BASE_A[k].name[0]);
        }
        
    }

    int BB_BONDS[][2] = {
        {0, 1}, {0, 2}, {0, 3},
        {3, 4},                
        {4, 5},                
        {5, 6},
        {5, 7},  
        {6, 10}, 
        {7, 8},
        {7, 9},
        {9, 10} 
    };
    int BB_BOND_COUNT = 11;

    int BASE_BONDS[][2] = {
        {0, 1}, {0, 9}, 
        {1, 2},         
        {2, 3},        
        {3, 4},
        {3, 9}, 
        {4, 5},
        {4, 6}, 
        {6, 7},
        {7, 8},
        {8, 9} 
    };
    int BASE_BOND_COUNT = 11;

    int atoms_per_res = BB_COUNT + BASE_A_COUNT;

    for (int i = 0; i < n; i++)
    {
        int base_id = 1 + i * atoms_per_res;

        for (int b = 0; b < BB_BOND_COUNT; b++)
        {
            fprintf(f, "CONECT%5d%5d\n", base_id + BB_BONDS[b][0], base_id + BB_BONDS[b][1]);
        }

        for (int b = 0; b < BASE_BOND_COUNT; b++)
        {
            fprintf(f, "CONECT%5d%5d\n", base_id + BB_COUNT + BASE_BONDS[b][0], base_id + BB_COUNT + BASE_BONDS[b][1]);
        }

        fprintf(f, "CONECT%5d%5d\n", base_id + 10, base_id + 11);

        if (i < n - 1)
        {
            int o3_id = base_id + 8;
            int next_p_id = base_id + atoms_per_res;
            fprintf(f, "CONECT%5d%5d\n", o3_id, next_p_id);
        }
    }
    fprintf(f, "END");


    fclose(f);
    printf("Structure saved to %s\n", filename);
}

float compute_potential_energy(int n, Vec3 *coords, Pair *pairs, int pair_count)
{
    float energy = 0.0f;

    for (int i = 0; i < n - 1; i++)
    {
        float dist = vec_norm(vec_sub(coords[i + 1], coords[i]));
        energy += 0.5f * K_BOND * (dist - BACKBONE_DIST) * (dist - BACKBONE_DIST);
    }

    for (int p = 0; p < pair_count; p++)
    {
        float dist = vec_norm(vec_sub(coords[pairs[p].j], coords[pairs[p].i]));
        energy += 0.5f * K_BOND * (dist - BASE_PAIR_DIST) * (dist - BASE_PAIR_DIST);
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            float dist = vec_norm(vec_sub(coords[j], coords[i]));
            if (dist < REPULSION_DIST && dist > 1e-6f)
            {
                energy += 0.5f * K_REPEL * (REPULSION_DIST - dist) * (REPULSION_DIST - dist);
            }
        }
    }
    return energy;
}

float compute_radius_of_gyration(int n, Vec3 *coords)
{
    Vec3 center = {0, 0, 0};
    for (int i = 0; i < n; i++)
        center = vec_add(center, coords[i]);
    center = vec_scale(center, 1.0f / n);

    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++)
    {
        Vec3 d = vec_sub(coords[i], center);
        sum_sq += vec_dot(d, d); /* |d|^2 */
    }
    return sqrtf(sum_sq / n);
}

void print_sequence_stats(const char *seq, int n, Pair *pairs, int pair_count)
{
    int counts[4] = {0};
    for (int i = 0; i < n; i++)
    {
        if (seq[i] == 'A')
            counts[0]++;
        else if (seq[i] == 'C')
            counts[1]++;
        else if (seq[i] == 'G')
            counts[2]++;
        else if (seq[i] == 'U')
            counts[3]++;
    }

    int pair_types[3] = {0};
    for (int p = 0; p < pair_count; p++)
    {
        char a = seq[pairs[p].i];
        char b = seq[pairs[p].j];
        if ((a == 'G' && b == 'C') || (a == 'C' && b == 'G'))
            pair_types[0]++;
        else if ((a == 'A' && b == 'U') || (a == 'U' && b == 'A'))
            pair_types[1]++;
        else
            pair_types[2]++;
    }

    printf("\n--- Sequence Analysis ---\n");
    printf("Length: %d\n", n);
    printf("GC Content: %.1f%%\n", (counts[1] + counts[2]) * 100.0f / n);
    printf("Base Pairs: %d\n", pair_count);
    printf("  GC: %d\n", pair_types[0]);
    printf("  AU: %d\n", pair_types[1]);
    printf("  GU: %d\n", pair_types[2]);
}

void rotate_base_template()
{
    for (int k = 0; k < BASE_A_COUNT; k++)
    {
        float old_x = BASE_A[k].pos.x;
        float old_y = BASE_A[k].pos.y;
        BASE_A[k].pos.x = old_y;
        BASE_A[k].pos.y = -old_x;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage: %s <sequence> [unused]\n", argv[0]);
        return 1;
    }

    rotate_base_template();

    char *seq = argv[1];
    int n = strlen(seq);

    // Use Linux, or else i'll break in your house and forcefully convert you to Linux
    struct stat st = {0};
    if (stat("PDB", &st) == -1)
    {
        mkdir("PDB", 0777);
    }

    char out_file[256];
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    strftime(out_file, sizeof(out_file) - 1, "PDB/output_%Y%m%d_%H%M%S.pdb", t);

    printf("Folding Sequence: %s (Len: %d)\n", seq, n);
    printf("Output will be saved to: %s\n", out_file);

    double start_total = omp_get_wtime();

    double start_2d = omp_get_wtime();
    int pair_count = 0;
    Pair *pairs = predict_structure(seq, &pair_count);
    double end_2d = omp_get_wtime();

    print_sequence_stats(seq, n, pairs, pair_count);

    double start_3d_init = omp_get_wtime();
    Vec3 *coords = malloc(n * sizeof(Vec3));
    for (int i = 0; i < n; i++)
    {
        float angle = i * 0.5f;
        float r = 10.0f;
        coords[i] = (Vec3){
            r * cosf(angle),
            r * sinf(angle),
            i * 0.5f};
    }
    double end_3d_init = omp_get_wtime();

    float init_energy = compute_potential_energy(n, coords, pairs, pair_count);
    float init_rg = compute_radius_of_gyration(n, coords);

    double start_physics = omp_get_wtime();
    physics_fold(n, coords, pairs, pair_count);
    double end_physics = omp_get_wtime();

    double start_save = omp_get_wtime();
    save_pdb(out_file, n, seq, coords);
    double end_save = omp_get_wtime();

    float final_energy = compute_potential_energy(n, coords, pairs, pair_count);
    float final_rg = compute_radius_of_gyration(n, coords);

    double end_total = omp_get_wtime();
    double time_total = end_total - start_total;
    double time_2d = end_2d - start_2d;
    double time_3d_init = end_3d_init - start_3d_init;
    double time_physics = end_physics - start_physics;
    double time_save = end_save - start_save;

    printf("\n--- Physics Stats ---\n");
    printf("Initial Potential Energy: %.2f kcal/mol\n", init_energy);
    printf("Final Potential Energy:   %.2f kcal/mol\n", final_energy);
    printf("Energy Reduction:         %.2f%%\n", (1.0f - final_energy / init_energy) * 100.0f);
    printf("Initial Radius of Gyration: %.2f Angstrom\n", init_rg);
    printf("Final Radius of Gyration:   %.2f Angstrom\n", final_rg);
    printf("Iterations / sec:         %.2f M/s\n", (ITERATIONS / 1e6) / time_physics);

    printf("\n--- Execution Time Analysis ---\n");
    printf("%-15s | %-10s | %-6s | %s\n", "Stage", "Time (s)", "%", "Graph");
    printf("------------------------------------------------------------\n");

    struct
    {
        char *name;
        double time;
    } stages[] = {
        {"2D Prediction", time_2d},
        {"3D Init", time_3d_init},
        {"Physics Fold", time_physics},
        {"PDB Output", time_save}};

    for (int i = 0; i < 4; i++)
    {
        double pct = (stages[i].time / time_total) * 100.0;
        int bars = (int)(pct / 2.0);
        if (bars == 0 && pct > 0)
            bars = 1;
        if (bars > 50)
            bars = 50;

        char bar_str[52];
        memset(bar_str, '#', bars);
        bar_str[bars] = '\0';

        printf("%-15s | %10.5f | %5.1f%% | %s\n", stages[i].name, stages[i].time, pct, bar_str);
    }
    printf("------------------------------------------------------------\n");
    printf("%-15s | %10.5f | %5.1f%% |\n", "Total", time_total, 100.0);

    free(pairs);
    free(coords);
    return 0;
}