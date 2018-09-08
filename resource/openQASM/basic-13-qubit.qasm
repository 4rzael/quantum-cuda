OPENQASM 2.0;
include "qelib1.inc";

qreg q[13];
creg c[13];

h q[0];
measure q[0] -> c[0];
h q[0];
measure q[0] -> c[0];
h q[0];
measure q[0] -> c[0];
x q[0];
measure q[0] -> c[0];
h q[0];
measure q[0] -> c[0];
h q[0];
measure q[0] -> c[0];
h q[0];
measure q[0] -> c[0];
h q[0];
measure q[0] -> c[0];
h q[0];
measure q[0] -> c[0];
h q[0];
measure q[0] -> c[0];
