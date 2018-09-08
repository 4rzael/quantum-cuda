OPENQASM 2.0;
include "qelib1.inc";

qreg q[12];
creg c[12];

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
h q[0];
measure q[0] -> c[0];
h q[0];
measure q[0] -> c[0];
h q[0];
measure q[0] -> c[0];
h q[0];
measure q[0] -> c[0];
