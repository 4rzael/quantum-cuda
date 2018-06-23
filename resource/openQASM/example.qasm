OPENQASM 2.0;
include "coucou.g";
qreg ninja[2]; // bla
creg bla[2];
U(pi/2, 0, pi) ninja[0];
CX ninja[0], ninja[1];

measure ninja -> bla;
//measure ninja[1] -> bla[2];

//barrier ninja, bla;
//if (bla == 5) creg pizza[1];
//mygate(1, 2, 3) q;
//mygate() q1, q2;
//mygate q;

//gate testgate(a) qa, qb {
//    CX qa, qb;
//}
