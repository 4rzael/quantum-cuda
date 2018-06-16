/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-16T09:36:50+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CPUExecutor.h
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-16T09:47:54+01:00
 * @License: MIT License
 */

#include "Executor.h"

 typedef std::valarray<std::complex<double>> Tvcplxd;

 class CPUExecutor : public Executor
 {
 public:
     CPUExecutor(){}
     ~CPUExecutor(){}
     Tvcplxd dot(Tvcplxd a, Tvcplxd b, int ma, int mb, int na, int nb);
     Tvcplxd kron(Tvcplxd a, Tvcplxd b, int ma, int mb);
     std::complex<double> tr(Tvcplxd a, int m);
     Tvcplxd T(Tvcplxd a, int m, int n);
 };
