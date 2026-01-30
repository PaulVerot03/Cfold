# We need a catchy name for this
## What this program does is :
1) Computes Secondary Structure using the Nussinov algorithm 
2) Computes Tertiary Structure based on molecular properties 
3) Exports to a PDB database file

## How this program works : 


/!\ To preface, I am neither a good coder or a bio student, I know surface level stuff and the great most of the following logic comes from academic papers and sometimes AI. 

BTW none of this will work on windows, it might work on Mac, but just use Linux like any sensible human being

All relevant sources and academic literature may be found in the .bib file 

TBD : link specific publication to methods in code and comments

### 1 | Nussinov Algorithm 
based on : 
https://rna.informatik.uni-freiburg.de/Teaching/index.jsp?toolName=Nussinov
https://math.mit.edu/classes/18.417/Slides/rna-prediction-zuker.pdf
https://users.cs.duke.edu/~brd/Teaching/Previous/Bio/2000/New-Readings/rivas-eddy-jmb.pdf
https://arxiv.org/pdf/1612.01639
`nussinov_compute(const char* seq, int n, bool* mask)`
Basic functioning : 
makes a square matrix with the sequence :
$\begin{pmatrix} D &X &C &U &U &A &C\\ C &0 &0 &0 &0 &1 &1 \\ U &X &0 &0 &0 &1 &1 \\ U &X &X &0 &0 &0 &0 \\ A &X &X &X &0 &0 &0 \\ C &X &X &X &X &0 &0 \end{pmatrix}$
D being the max amount of pairs that two given nucleotide can make
For each cell $D_(i,j)$ : 
- if i and j make a pair then we'll take the max value of (D(i,k-1)+D(k+1, j-1)+1) if K and J make a pair for k in j - seq_len and i. Basically, if we encounter a pair, then we backtrack to see if other pairs form. 
- if i and j do not make a pair, then we take D(i,j-i)
And then cell value is updated to the max of those two.

MFE can be computed to evaluate best structure. *TBD*

In this implementation, complexity is about O(n^3)

The function returns a list of (i,j) base pair indices

### 2 | Preliminary 3D structure 
We compute the secondary structure using the nussinov library, which returns a list of pairs. 
Then we put those pairs in a sort of helix. The reason for this that I got better structure when doing this. Which may be a hint to some inneficiencies in the code some place else. 
Without such modification, structure remains "flat". This part of the code needs to be looked at more carefuly as I suppose the assumption I am making here may lead to innacuracies.

Then we can refine the structure using the physics fold function


### 3 | Refining 3D structure
So this will do in a shit load of iterations the following : 
- compute forces → bond, angle, base pairs and repulsion
	- repulsion :  because electrical energy of the atoms attract and repel each other, so we approximate that (base pair attract, all atoms repel non bonded neighbours )
	- angle : each pair can't be twisted to much. it has wiggle room so i guess this is our biggest source of inaccuracy/guess because this might be constrained by way to many external factors we just can't represent in any way (type of shit Ian Malcolm from Jurassic Park would argue) papers suggest this should be upper-bounded by 2 rad°
	- bond : "mechanical" push/pull, idk the level of accuracy of this, we might be wayyyy off. I kinda just guessed. From what i gathered, it's like a pearl necklace, the beads just exert force on their neighbours 
- applies those forces → just updates the array entry
- update position → pure hell of linear algebra and Euler wizardry  
- decrease temp → simulates annealing 
	- this is basically a linear program that tries to minimise $E(bond)+E(base pair) +E(repulsion)$ while $\begin{cases} E(bond) = \sum \frac{K\_BOND \times (d_i - d_0)²}{2} \\ E(base\_pair) = \sum \frac{5 \times K\_BOND \times (d\_bp - d\_bp_0)²}{2} \\ E(repulsion) = \sum \frac{K\_REPEL \times (d_repul - d_{ij})²}{2} \text{ for } d_{ij} < d\_repul \end{cases}$
Basically i'm trying to get an MFE
Minimizing energy leads to a more stable structure because we want to minimize potential energy. This is like a coil spring from a car, when handleing it, we want it to have a minimum of energy. 


```C
K_BOND = 1.0 // (kcal/mol/Å²)
K_REPEL = 1.0 //  (kcal/mol/Å²)
BACKBONE_DIST = 6.0 // Ideal P-to-P spacing (Å)
BASE_PAIR_DIST = 10.0 // Ideal Watson-Crick distance (Å)
REPULSION_DIST = 12.0 // Excluded volume cutoff (Å)
ANGLE_TARGET = 2.0 // Ideal C-C-C angle (rad ≈ 115°)
LEARNING_RATE = 0.2 // Initial step size (Å)
ITERATIONS = 50000 // The good thing is that this is only linear
TEMPERATURE_DECAY = 0.2// Simulated annealing rate
```
This is the biggest guess work, i'm sure we need to tweak the values as they are mostly arbitrary

This function has the biggest cost @ O(ITERATIONS * (3 n + (seq_len * n^2))) 

There is 2 implementation of this logic, one that runs in serial for short sequences and one that runs in parallel. 
`physics_fold_serial(int n, Vec3* coords, Pair* pairs, int pair_count)`
`physics_fold_parallel(int n, Vec3* coords, Pair* pairs, int pair_count)`
because for short sequence, we loose more time initializing the paralelization than if we just bruteforced the sequence in mono-thread

Each thread does 25 atoms, we might have a problem with that. Meaning the cutoff might reduce accurary. It's probbaly worth it to toy with the value.

### 4 | 3D Representation 
Now we need to take the array with all the updated coordinates and map it. 
I let Gemini do that part because I am bad at 3D 
I don't get how it works really 

Just read the comment it left : 
```txt
/* First nucleotide: use arbitrary perpendicular via cross product */
/* General case: bisector of incoming and outgoing directions */
/* Gram-Schmidt orthogonalization: bisector - (bisector·T)*T */
```
What in the ever-loving fuck is this shit ?

Saving to PDB is just a matter of reading the documentation for the format and printing with the correct line heading.
PDB are saved to the PDB folder (creates it if !found)
Documentation for PDB is shite so we may have weird stuff going on

### 5 | Stats
Because we want to know additional infos, we need to compute some statistics
So we do energy and gyration (distance to center, so roughly half the diameter)

And then we have `main()` which just does launches the calculation and prints stats about computation time

## Running this program

I use a Python wrapper that compiles and run the c file, and print compile and compute time. Set sequence in the `run.py` file.
`python3 run.py Cfold.c "comment for the log file"`, i forgot how to make an argument optional , so you have to include a comment, i could look this up, or i can write a comment each time

Then go brew yourself a coffee because this will take a little while. It took me about 13 min for a 3000 nucleotide sequence
I may try to implement a time estimation based on the computer's hardware and sequence lenght and iterations, but I don't know how accurate that could be.
I guess I just need infos from /proc/cpu* and that could be done in the python wrapper if in can somehow read iteration number from the C file in python, or pass it as an argument ?  

## References 
https://users.cs.duke.edu/~brd/Teaching/Previous/Bio/2000/New-Readings/rivas-eddy-jmb.pdf
https://math.mit.edu/classes/18.417/Slides/rna-prediction-zuker.pdf
https://epubs.siam.org/doi/10.1137/0135006
https://rna.informatik.uni-freiburg.de/Teaching/index.jsp?toolName=Nussinov
https://arxiv.org/pdf/1612.01639 
https://web.azenta.com/hubfs/2022-03%20GEN%20NGS%20-%20Guide%20to%20RNASeq%20eBook/13002-WE%200222%20RNA-Seq%20E-Book.pdf

1. **Nussinov & Jacobson (1980)**: "Fast algorithm for predicting the secondary structure of single-stranded RNA". *Proceedings of the National Academy of Sciences USA*, 77(11), 6309-6313.
   - Foundational DP algorithm

2. **Rivas & Eddy (1999)**: "A dynamic programming algorithm for RNA structure prediction including pseudoknots". *Journal of Molecular Biology*, 285(5), 2053-2068.
   - Pseudoknot extensions

3. **Cornell et al. (1995)**: "A second generation force field for the simulation of proteins, nucleic acids, and organic molecules". *Journal of the American Chemical Society*, 117(19), 5179-5197.
   - AMBER force field parameters

4. **Kirkpatrick et al. (1983)**: "Optimization by simulated annealing". *Science*, 220(4598), 671-680.
   - Simulated annealing theory

5. **Saenger, W. (1984)**: *Principles of Nucleic Acid Structure*. Springer-Verlag.
   - Comprehensive RNA chemistry and geometry

6. **Berman et al. (2000)**: "The Protein Data Bank". *Nucleic Acids Research*, 28(1), 235-242.
   - PDB format and structure database

7. **Drew et al. (1981)**: "Structure of a B-DNA dodecamer". *Proceedings of the National Academy of Sciences USA*, 78(4), 2179-2183.
   - B-form nucleic acid structure

8. **Xia et al. (1998)**: "Thermodynamic parameters for an expanded nearest-neighbor model for formation of RNA duplexes". *Biochemistry*, 37(42), 14719-14735.
   - Base pair energy values

9. **Crick, F.H. (1966)**: "Codon–anticodon pairing: The wobble hypothesis". *Journal of Molecular Biology*, 19(2), 548-555.
   - Wobble base pairing explanation

10. **Watson & Crick (1953)**: "Molecular structure of nucleic acids: a structure for deoxyribose nucleic acid". *Nature*, 171(4356), 737-738.
    - Original Watson-Crick base pairing
