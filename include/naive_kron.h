/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:04:59+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: naive_kron.h
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-15T11:21:24+01:00
 * @License: MIT License
 */

#pragma once

#include <valarray>
#include <complex>

typedef std::valarray<std::complex<double>> Tvcplxd;

Tvcplxd naive_kron(Tvcplxd a, Tvcplxd b, int ma, int mb);
