OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

h q[0];
barrier q;
cx q[0], q[1];
h q[2];

measure q -> c;
