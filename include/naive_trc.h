/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:58:11+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: naive_mult.h
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-13T13:59:21+01:00
 * @License: MIT License
 */

#pragma once

#include <valarray>
#include <complex>

typedef std::valarray<std::complex<double>> Tvcplxd;

std::complex<double> trc(Tvcplxd a, int m);
