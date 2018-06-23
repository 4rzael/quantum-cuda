OPENQASM 2.0;

// Name of Experiment: Deutsch Algorithm v3
// Description: Simple two qubit implementation of Deutsch algorithm. It shows that, for a given function f, in order to know value of f(0) xor f(1) can be computed using 1 query. If result is 1, f is balanced, otherwise it is constant funcion

gate cx a, b { CX a, b; }
gate x a { U(pi,0,pi) a; }
gate h a { U(pi/2, 0,pi) a; }

qreg q[2];
creg c[1];

// input preparation
x q[1];
h q[0];
h q[1];
// f
cx q[0],q[1];
// output preparation
h q[0];
// measurement
measure q[0] -> c[0];