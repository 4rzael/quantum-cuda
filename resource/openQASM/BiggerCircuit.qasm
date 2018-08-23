OPENQASM 2.0;
include "qelib1.inc";

qreg qh[5];
qreg qx[5];

qreg howbigcanwego[2];

creg ch[5];
creg cx[5];

h qh;
cx qh, qx;

measure qh -> ch;
measure qx -> cx;
