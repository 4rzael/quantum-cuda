# quantum-cuda
CUDA-based Simulator of Quantum Systems

## Project Overview
Simulating quantum systems is computationally hard. Generally speaking,
the resource cost (time complexity) of simulating a quantum system grows
exponentially in the size of said system. However, there are many tools that
have been developed to tackle a small set of special cases. Also, there are
analytical tools that can reduce the cost of simulation for a larger set of
cases. Finally, many advances in numerical simulation have been developed in
recent years that take direct advantage of GPUs and their vastly superior
computational powers when compared to CPUs specifically in the area of
linear dynamic systems. This project will attempt to synthesise and exploit all
the above tools to develop a simulation tool that is usefully more efficient
than existing ones.

## Repository Content
The content of this repository, in accordance with our supervisor's
expectations, constitutes an early-stage prototype of the final corpus of the
project, namely: a working, but not optimised simulator of quantum systems able
to run on simple circuits defined following the openQASM syntax.

## Requirements
To be able to compile the project on a given computer GCC should be installed as
well as CUDA and Boost libraries.
Fulfilling compilation requirements could be cumbersome at this stage and
proper environment deployment instructions should be defined in the future.
At the moment running the produced binary would require a compatible GPU to run
linear algebra computations on. It will soon be possible to automatically detect
the absence of compatible GPU and compute all linear algebra on the CPU.
