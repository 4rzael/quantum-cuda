OPENQASM 2.0;
include "coucou.g";
qreg ninja[3]; // bla
creg bla[2];
CX bla, ninja[2];
U(0,pi/4, ((3/4)*(pi/2))) bla[0];

measure ninja -> bla;
measure ninja[1] -> bla[2];

barrier ninja, bla;
if (bla == 5) creg pizza[1];
mygate(1, 2, 3) q;
mygate() q1, q2;
mygate q;

gate testgate(a) qa, qb {
    CX qa, qb;
}
