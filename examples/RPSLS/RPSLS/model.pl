nn(rpsls_net,[X],Y,[paper,scissors,rock,lizard,spock]) :: sign(X,Y).

rpsls(X,Y,0) :- sign(X,Z), sign(Y,Z).

rpsls(X,Y,1) :- sign(X,paper), sign(Y,rock).
rpsls(X,Y,2) :- sign(X,paper), sign(Y,scissors).
rpsls(X,Y,2) :- sign(X,paper), sign(Y,lizard).
rpsls(X,Y,1) :- sign(X,paper), sign(Y,spock).

rpsls(X,Y,1) :- sign(X,scissors), sign(Y,paper).
rpsls(X,Y,2) :- sign(X,scissors), sign(Y,rock).
rpsls(X,Y,1) :- sign(X,scissors), sign(Y,lizard).
rpsls(X,Y,2) :- sign(X,scissors), sign(Y,spock).

rpsls(X,Y,1) :- sign(X,rock), sign(Y,scissors).
rpsls(X,Y,2) :- sign(X,rock), sign(Y,paper).
rpsls(X,Y,1) :- sign(X,rock), sign(Y,lizard).
rpsls(X,Y,2) :- sign(X,rock), sign(Y,spock).

rpsls(X,Y,2) :- sign(X,lizard), sign(Y,scissors).
rpsls(X,Y,1) :- sign(X,lizard), sign(Y,paper).
rpsls(X,Y,2) :- sign(X,lizard), sign(Y,rock).
rpsls(X,Y,1) :- sign(X,lizard), sign(Y,spock).

rpsls(X,Y,1) :- sign(X,spock), sign(Y,scissors).
rpsls(X,Y,2) :- sign(X,spock), sign(Y,paper).
rpsls(X,Y,1) :- sign(X,spock), sign(Y,rock).
rpsls(X,Y,2) :- sign(X,spock), sign(Y,lizard).