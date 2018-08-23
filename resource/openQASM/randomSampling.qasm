OPENQASM 2.0;
include "qelib1.inc";

qreg qh[5];
creg ch[5];

h qh;

measure qh -> ch;
