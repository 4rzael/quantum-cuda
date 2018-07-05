OPENQASM 2.0;
include "qelib1.inc";
qreg ninja[2]; // bla
creg bla[2];
U(pi/2, 0, pi) ninja[0];
CX ninja[0], ninja[1];

measure ninja -> bla;
//measure ninja[1] -> bla[2];

barrier ninja;
if (bla == 5) U(0,0,0) ninja;

reset ninja;

gate mygate1(a) qa, qb {
    CX qa, qb;
    U(0,pi/4, ((a/4)*(pi/2))) qa;
}

gate mygate2(ah) bla1, bla2 {
    mygate1(ah * 4) bla1, bla2;
}


mygate2(100) ninja[0], ninja[1];
//mygate() q1, q2;
//mygate q;
