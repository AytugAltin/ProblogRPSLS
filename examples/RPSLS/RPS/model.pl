nn(rps_net,[X],Y,[paper,scissors,rock]) :: sign(X,Y).

rps(X,Y,0) :- sign(X,paper), sign(Y,paper).
rps(X,Y,0) :- sign(X,rock), sign(Y,rock).
rps(X,Y,0) :- sign(X,scissors), sign(Y,scissors).

rps(X,Y,1) :- sign(X,paper), sign(Y,rock).
rps(X,Y,2) :- sign(X,paper), sign(Y,scissors).

rps(X,Y,2) :- sign(X,rock), sign(Y,paper).
rps(X,Y,1) :- sign(X,rock), sign(Y,scissors).

rps(X,Y,1) :- sign(X,scissors), sign(Y,paper).
rps(X,Y,2) :- sign(X,scissors), sign(Y,rock).