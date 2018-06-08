#include <iostream>
#include <cmath>

#include "build_gate.h"

int main(int ac, char **av) {
    // I = th=0, ph=0, l=0
    cdmatrix i = build_gate(0, 0, 0);
    printf("\nI = \t[[%.2f+%.2fi, %.2f+%.2fi],\n\t[%.2f+%.2fi, %.2f+%.2fi]]\n",
      i[0].real(), i[0].imag(),
      i[1].real(), i[1].imag(),
      i[2].real(), i[2].imag(),
      i[3].real(), i[3].imag());
    // Pauli-X = th=M_PI, ph=0, l=M_PI
    cdmatrix x = build_gate(M_PI, 0, M_PI);
    printf("\nX = \t[[%.2f+%.2fi, %.2f+%.2fi],\n\t[%.2f+%.2fi, %.2f+%.2fi]]\n",
      x[0].real(), x[0].imag(),
      x[1].real(), x[1].imag(),
      x[2].real(), x[2].imag(),
      x[3].real(), x[3].imag());
    // Pauli-Y = th=M_PI, ph=M_PI/2, l=M_PI/2
    cdmatrix y = build_gate(M_PI, M_PI/2, M_PI/2);
    printf("\nY = \t[[%.2f+%.2fi, %.2f+%.2fi],\n\t[%.2f+%.2fi, %.2f+%.2fi]]\n",
      y[0].real(), y[0].imag(),
      y[1].real(), y[1].imag(),
      y[2].real(), y[2].imag(),
      y[3].real(), y[3].imag());
    // Pauli-Z = th=0, ph=0, l=M_PI
    cdmatrix z = build_gate(0, 0, M_PI);
    printf("\nZ = \t[[%.2f+%.2fi, %.2f+%.2fi],\n\t[%.2f+%.2fi, %.2f+%.2fi]]\n",
      z[0].real(), z[0].imag(),
      z[1].real(), z[1].imag(),
      z[2].real(), z[2].imag(),
      z[3].real(), z[3].imag());
    // Hadamard = th=M_PI/2, ph=0, l=M_PI
    cdmatrix h = build_gate(M_PI/2, 0, M_PI);
    printf("\nH = \t[[%.2f+%.2fi, %.2f+%.2fi],\n\t[%.2f+%.2fi, %.2f+%.2fi]]\n",
      h[0].real(), h[0].imag(),
      h[1].real(), h[1].imag(),
      h[2].real(), h[2].imag(),
      h[3].real(), h[3].imag());
    // Phase = th=0, ph=0, l=M_PI/2
    cdmatrix s = build_gate(0, 0, M_PI/2);
    printf("\nS = \t[[%.2f+%.2fi, %.2f+%.2fi],\n\t[%.2f+%.2fi, %.2f+%.2fi]]\n\n",
        s[0].real(), s[0].imag(),
        s[1].real(), s[1].imag(),
        s[2].real(), s[2].imag(),
        s[3].real(), s[3].imag());
    return EXIT_SUCCESS;
}
