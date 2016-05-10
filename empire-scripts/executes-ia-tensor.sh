#!/usr/bin/bash

SPORT=$(( ( RANDOM % 1000 ) + 2000 ))

IA1=$1

if test x$IA1 = x; then
	echo "usage: $0 <ia1>"
	exit 1
fi

source ./execute-lib.sh ; cd ..
make_empire

NBWIN0=0
NBWIN1=0

NB=25
PART=0
while test $NB -gt 0; do
	echo "Partie $PART démarrée"
	NB=`expr $NB - 1`
	PART=`expr $PART + 1`
	# Demarrage des programmes.
	./empire-server/Main.native -sport ${SPORT}  > out_S 2>&1 & 
	SPID=$!
	sleep 1
	python ./empire-captain/ai${IA1}.py localhost ${SPORT} > out_P1 2>&1 &
	PPID1=$!
	python ./DeepLearningGame_TensorFlow/main.py localhost ${SPORT} &
	PPID2=$!

	PIDS="${SPID} ${PPID1} ${PPID2}"

	# Regarde si un des programmes est stoppe
	STOPPED=0
	while test $STOPPED -eq 0; do
		sleep 2
		check_processes STOPPED ${PIDS}
	done

	# Arret de tous les programmes.
	stop_processes ${PIDS}

	tail -n 1 out_S | grep 'winner 0' > /dev/null
	if test $? -eq 0; then
		NBWIN0=`expr $NBWIN0 + 1`
	else
		tail -n 1 out_S | grep 'winner 1' > /dev/null
		if test $? -eq 0; then
			NBWIN1=`expr $NBWIN1 + 1`
		else
			echo "ERR!"
			exit
		fi
	fi
	echo "END PART $PART : $NBWIN0 $NBWIN1"
	echo "Player0 : $NBWIN0" > ./resultat
	echo "Player1 : $NBWIN1" >> ./resultat
done
echo $NBWIN0 $NBWIN1
