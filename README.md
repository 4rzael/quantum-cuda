# quantum-cuda

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/492cb830738c4c58881e2fd2b34fee12)](https://app.codacy.com/app/vial-dj/quantum-cuda?utm_source=github.com&utm_medium=referral&utm_content=4rzael/quantum-cuda&utm_campaign=Badge_Grade_Dashboard)

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
Compiling the project requires a computer running linux, with a c++14 compiler
such as GCC or CLANG, as well as CUDA and Boost libraries.
Fulfilling compilation requirements could be cumbersome at this stage and
proper environment deployment instructions should be defined in the future.
At the moment running the produced binary would require a compatible GPU to run
linear algebra computations on. It will soon be possible to automatically detect
the absence of compatible GPU and compute all linear algebra on the CPU.

## Other informations

More informations are available on [the wiki](https://github.com/4rzael/quantum-cuda/wiki), including the presentation of the general architecture of the project. We also use doxygen for a more in-depth documentation.
