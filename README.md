# Parallel Logistic Regression

Sequential, OpenMP, and MPI binary logistic regression. Sparse SVMlight input, full-batch gradient descent, L2 regularization.

Compiler: GCC 12.3.0, C++17, `-O3 -std=c++17 -Wall -Wextra`, OpenMP adds `-fopenmp`, MPI uses `mpic++` / OpenMPI 4.1.5.

## unitn VPN + MobaX

I run everything on the unitn HPC cluster (`hpc.unitn.it`) from Windows using VPN and MobaXterm.

### Step 1  Open GlobalProtect VPN and log in to connect.

### Step 2  IN MobaXterm using host `hpc.unitn.it`, username is my cluster account.

### Step 3  I drag the directory into the left panel

### Step 4  Go to project directory

In the MobaXterm terminal:

```bash
cd ~/PLR_submission
pwd
ls
```
I should see `src`, `scripts`, `data`, `results`, etc.

### Step 5 Load and compile

Still in terminal:

```bash
module purge
module load GCC/12.3.0
module load OpenMPI/4.1.5-GCC-12.3.0
module list
gcc --version
mpic++ --version
cd src
make
cd ..
```

This builds `src/seq`, `src/omp`, `src/mpi`.


### Step 6  Submit all experiments

From root directory,

(if problem occurs)
```bash
dos2unix scripts/go_seq.pbs scripts/go_mpi.pbs scripts/go_omp.pbs
```

Then submit all three:

```bash
qsub scripts/go_seq.pbs
qsub scripts/go_mpi.pbs
qsub scripts/go_omp.pbs
```

`go_seq.pbs` uses queue **`shortCPUQ`**, `#PBS -l select=1:ncpus=1:mpiprocs=1` (1 core). Logs go to `results/`; PBS stdout/stderr go to `seq_1.o<JOBID>` and `seq_1.e<JOBID>`. `results/SUMMARY_seq.txt` is written at the end.

`go_mpi.pbs` uses queue **`shortCPUQ`**, `#PBS -l select=16:ncpus=4:mpiprocs=4` (16 nodes, 64 CPUs). MPI jobs use an Open MPI hostfile built from `$PBS_NODEFILE`. Logs go to `results/`; PBS stdout/stderr go to `mpi_64.o<JOBID>` and `mpi_64.e<JOBID>`. `results/SUMMARY_mpi.txt` is written at the end.

`go_omp.pbs` uses queue **`shortCPUQ`**, `#PBS -l select=1:ncpus=64:mpiprocs=1` (1 node, 64 CPUs). Logs go to `results/`; PBS stdout/stderr go to `omp_64.o<JOBID>` and `omp_64.e<JOBID>`. `results/SUMMARY_omp.txt` is written at the end.

### Step 7  Check job status

```bash
qstat -u $USER
```

(`$USER` is my username same as in `username@hpc.unitn.it`.)

Q = queued, R = running, gone from list = finished (or failed).

When finished,

```bash
ls seq_1.o* seq_1.e* mpi_64.o* mpi_64.e* omp_64.o* omp_64.e* 2>/dev/null
ls results/*.log | wc -l
cat results/SUMMARY_seq.txt
cat results/SUMMARY_mpi.txt
cat results/SUMMARY_omp.txt
tail -2 results/gisette_mpi_64.log
tail -2 results/gisette_omp_64.log
```

Expect `Training finished` in `gisette_mpi_64.log` without `not enough slots`, and `OpenMP threads used: 64` in `gisette_omp_64.log`.

### Step 8 Download results
for the results, right-click Download or drag to local folder.

Same download `results/SUMMARY_seq.txt`, `results/SUMMARY_mpi.txt`, `results/SUMMARY_omp.txt`, or updated `data/` if needed.

## Data

| Dataset | Train | Test | Features |
|---------|-------|------|----------|
| Gisette | 4800 | 1200 | 5000 |
| 20News 2-class | 1177 | 783 | 130107 |
| 20News full | 11314 | 7532 | 130107 |

Gisette: `data/gisette/*.svm`. 20News: exported by scripts in `scripts/` (need `pip install --user scikit-learn`).

- 20NewsGroup 2-class (`comp.graphics` vs `sci.space`): `python3 scripts/export_20newsgroups_binary.py`
- 20NewsGroup full  (categories 0--9 vs 10--19): `python3 scripts/export_20newsgroups_full.py`

Both scripts accept optional flags such as `--out-dir`, `--prefix`, and (for the binary script) `--class-a`, `--class-b`, `--list-categories`. Defaults write to `data/20newsgroup/` and `data/20newsgroup_full/`.

## usage

```bash
cd src
./seq  <train.svm> <test.svm> <epochs> <lr> <lambda>
./omp  <train.svm> <test.svm> <epochs> <lr> <lambda> <threads>
mpirun -np N ./mpi <train.svm> <test.svm> <epochs> <lr> [lambda] [overlap]
```

Examples for all three datasets (run from `src/`):

```bash
cd src
./seq ../data/gisette/gisette_local_train.svm ../data/gisette/gisette_local_test.svm 100 0.01 0.01
./omp ../data/gisette/gisette_local_train.svm ../data/gisette/gisette_local_test.svm 100 0.01 0.01 4
mpirun -np 4 ./mpi ../data/gisette/gisette_local_train.svm ../data/gisette/gisette_local_test.svm 100 0.01 0.01 0

cd src
./seq ../data/20newsgroup/20newsgroup_train.svm ../data/20newsgroup/20newsgroup_test.svm 200 1.0 0.001
./omp ../data/20newsgroup/20newsgroup_train.svm ../data/20newsgroup/20newsgroup_test.svm 200 1.0 0.001 4
mpirun -np 4 ./mpi ../data/20newsgroup/20newsgroup_train.svm ../data/20newsgroup/20newsgroup_test.svm 200 1.0 0.001 0

cd src
./seq ../data/20newsgroup_full/20newsgroup_full_train.svm ../data/20newsgroup_full/20newsgroup_full_test.svm 200 1.0 0.001
./omp ../data/20newsgroup_full/20newsgroup_full_train.svm ../data/20newsgroup_full/20newsgroup_full_test.svm 200 1.0 0.001 4
mpirun -np 4 ./mpi ../data/20newsgroup_full/20newsgroup_full_train.svm ../data/20newsgroup_full/20newsgroup_full_test.svm 200 1.0 0.001 0
```

if need MPI with several nodes on the cluster, do not just run `mpirun -np 64 ./mpi ...` (fails with `not enough slots`), submit `qsub scripts/go_mpi.pbs`
which runs all MPI experiments and starts MPI on every node that PBS gives. Sequential baseline is in `go_seq.pbs`; OpenMP scaling is in `go_omp.pbs` (single node, 64 CPUs).

`overlap`: 0 = `MPI_Allreduce`, 1 = `MPI_Iallreduce`.

Defaults: Gisette 100 epochs, lr 0.01, lambda 0.01; 20News 200 epochs, lr 1.0, lambda 0.001.

in `go_omp.pbs`: OpenMP 1, 2, 4, 8, 16, 32, 64 threads. in `go_mpi.pbs`: MPI 1, 2, 4, 8, 16, 32, 64 processes (+ overlap). in `go_seq.pbs`: sequential baseline + repeats.

Output: one line per epoch with time, train/test loss and accuracy.

## Results

Log files in `results/`, `results/SUMMARY_seq.txt`, `results/SUMMARY_mpi.txt`, `results/SUMMARY_omp.txt`. Figures in `plots/`: `accuracy.png`, `speedup.png`, `efficiency.png`.

## Clean

```bash
cd src && make clean
```
