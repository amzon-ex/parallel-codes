# Parallel implementations of random algorithms

I'm learning MPI and hence I've made this repo to store parallel implementations of codes I've written. Codes are written in Python (unless otherwise indicated?) with `mpi4py`. Usually they will be accompanied with extra things like the serial implementation, datafiles containing sample output, additional codes to read that output and produce graphs (if relevant), profiling/scalability study.

Most of these will implement purely MPI but *might* include `numba` additives (to use JIT or thread-based parallelization: Python cannot use OpenMP).

I'm trying to achieve class-based implementations but it is tricky to make `numba` work with classes straightforwardly.
