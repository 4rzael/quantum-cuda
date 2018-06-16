/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-16T09:38:03+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: Executor.h
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-16T09:56:15+01:00
 * @License: MIT License
 */

#pragma once

#include <valarray>
#include <complex>

typedef std::valarray<std::complex<double>> Tvcplxd;

class Executor
{
  public:
    Executor(){}
    virtual ~Executor(){}
    virtual Tvcplxd dot(Tvcplxd a, Tvcplxd b, int ma, int mb, int na, int nb) = 0;
    virtual Tvcplxd kron(Tvcplxd a, Tvcplxd b, int ma, int mb) = 0;
    virtual std::complex<double> tr(Tvcplxd a, int m) = 0;
    virtual Tvcplxd T(Tvcplxd a, int m, int n) = 0;
};
