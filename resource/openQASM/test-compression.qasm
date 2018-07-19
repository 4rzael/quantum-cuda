OPENQASM 2.0;
include "qelib1.inc";
qreg ninja[3]; // bla
creg bla[2];
U(pi/2, 0, pi) ninja[0];
U(pi/2, 0, pi) ninja[1];
U(pi/2, 0, pi) ninja[2];

U(pi/2, 0, pi) ninja[0];
U(pi/2, 0, pi) ninja[0];
U(pi/2, 0, pi) ninja[0];

U(pi/2, 0, pi) ninja;

