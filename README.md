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
cd ~/PLR-submission
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

(if 20News `.svm` files are missed)

```bash
pip install --user scikit-learn
python3 scripts/export_20newsgroups_binary.py
python3 scripts/export_20newsgroups_full.py
```

### Step 6  Submit all experiments

From root directory,

(if problem occurs)
```bash
dos2unix scripts/go_all.pbs
```

Then submit:

```bash
qsub scripts/go_all.pbs
```

`go_all.pbs` uses queue `short_cpuQ`, `#PBS -l select=1:ncpus=8`, walltime 30 minutes. logs go to `results/`; PBS stdout/stderr go to `go_all.o<JOBID>` and `go_all.e<JOBID>`. `results/SUMMARY.txt` is written.

### Step 7  Check job status

```bash
qstat -u $USER
```

(`$USER` is my username same as in `username@hpc.unitn.it`.)

Q = queued, R = running, gone from list = finished (or failed).

When finished,

```bash
ls go_all.o* go_all.e* 2>/dev/null
ls results | wc -l
cat results/SUMMARY.txt
tail -2 results/gisette_seq.log
```

### Step 8 Download results
for the results, right-click Download or drag to local folder.

Same download `results/SUMMARY.txt` or updated `data/` if needed.

## Data

| Dataset | Train | Test | Features |
|---------|-------|------|----------|
| Gisette | 4800 | 1200 | 5000 |
| 20News 2-class | 1177 | 783 | 130107 |
| 20News full | 11314 | 7532 | 130107 |

Gisette: `data/gisette/*.svm`. 20News: exported by scripts in `scripts/` (need `scikit-learn` on the cluster: `pip install --user scikit-learn`).

- **20NewsGroup 2-class** (`comp.graphics` vs `sci.space`): `python3 scripts/export_20newsgroups_binary.py`
- **20NewsGroup full** (categories 0--9 vs 10--19): `python3 scripts/export_20newsgroups_full.py`

Both scripts accept optional flags such as `--out-dir`, `--prefix`, and (for the binary script) `--class-a`, `--class-b`, `--list-categories`. Defaults write to `data/20newsgroup/` and `data/20newsgroup_full/`.

## usage

```bash
cd src
./seq  <train.svm> <test.svm> <epochs> <lr> <lambda>
./omp  <train.svm> <test.svm> <epochs> <lr> <lambda> <threads>
mpirun -np N ./mpi <train.svm> <test.svm> <epochs> <lr> [lambda] [overlap]
```

(cd src
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
mpirun -np 4 ./mpi ../data/20newsgroup_full/20newsgroup_full_train.svm ../data/20newsgroup_full/20newsgroup_full_test.svm 200 1.0 0.001 0)

`overlap`: 0 = `MPI_Allreduce`, 1 = `MPI_Iallreduce`.

Defaults: Gisette 100 epochs, lr 0.01, lambda 0.01; 20News 200 epochs, lr 1.0, lambda 0.001. Thread/process counts 1, 2, 4, 8.

Output: one line per epoch with time, train/test loss and accuracy.


## Results

63 log files in `results/`, report figures in `plots/`: `accuracy.png`, `speedup.png`.

## Clean

```bash
cd src && make clean
```
