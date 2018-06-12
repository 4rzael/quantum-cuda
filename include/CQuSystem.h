/**
 * @Author: Julien Vial-Detambel <vial-d_j>
 * @Date:   2018-06-12T11:16:51+01:00
 * @Email:  julien.vial-detambel@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: CQuSystem.h
 * @Last modified by:   vial-d_j
 * @Last modified time: 2018-06-12T12:38:24+01:00
 * @License: MIT License
 */

#pragma once

#include <vector>
#include <complex>
#include <cstdint>

/** A convenient typedef for std::vector<std::complex<double>> */
typedef std::vector<std::complex<double>> Tvcplxd;

/**
* Quantum system representation class.
*/
class CQuSystem
{
    private:
      /**
      * The number of qubits in the system
      */
      uint32_t _muiSize;
      /**
       * A vector of complex double reprensenting the state.
       */
      Tvcplxd _mvcplxdState;

    public:
      /**
       * Construct a QuSystem object from a given size of qubits in state |0>.
       * @param uiSize The number of qubits to create.
       */
      CQuSystem(uint32_t uiSize);
      /**
       * State getter..
       * @return Return the quantum system state.
       */
      Tvcplxd getState();
      /**
       * Number of qubits in the system getter.
       * @return The number of qubits in the sytem.
       */
      uint32_t getSize();
      /*/**
       * Applies a tranformation matrix to the current state.
       * @param vcplxdTransformation A vector of complex double representing a
       *  kroenecker product of quantum gates.
      void Transform(Tvcplxd vcplxdTransformation);
      **/
};
