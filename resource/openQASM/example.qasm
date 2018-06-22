OPENQASM 2.0;
include "coucou.g";
qreg ninja[3]; // bla
creg bla[3];
CX ninja[1], ninja[2];
U(0,pi/4, ((3/4)*(pi/2))) ninja[0];

measure ninja -> bla;
measure ninja[1] -> bla[2];

barrier ninja, bla;
if (bla == 5) creg pizza[1];

gate mygate(a) qa, qb {
    CX qa, qb;
    U(0,pi/4, ((a/4)*(pi/2))) qa;
}


mygate(100) ninja[0], ninja[2];
//mygate() q1, q2;
//mygate q;
