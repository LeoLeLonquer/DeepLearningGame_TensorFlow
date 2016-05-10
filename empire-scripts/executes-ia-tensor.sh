#!/usr/bin/bash

SPORT=$(( ( RANDOM % 1000 ) + 2000 ))
PPORT1=$(( ( RANDOM % 1000 ) + 3000 ))
PPORT2=$(( ( RANDOM % 1000 ) + 4000 ))
OPORT1=$(( ( RANDOM % 1000 ) + 5000 ))
OPORT2=$(( ( RANDOM % 1000 ) + 6000 ))

IA1=$1

if test x$IA1 = x; then
	echo "usage: $0 <ia1>"
	exit 1
fi

source ./execute-lib.sh ; cd ..
make_empire

NBWIN0=0
NBWIN1=0

NB=2
while test $NB -gt 0; do
	NB=`expr $NB - 1`
	# Demarrage des programmes.
	launch_xterm "./empire-server/Main.native -sport ${SPORT} 2>&1 | tee out_S" SPID
	sleep 1
	launch_xterm "./empire-tee/tee.py localhost ${SPORT} ${PPORT1} ${OPORT1}" TPID1
	launch_xterm "./empire-client/Main.native -obs -sport ${OPORT1}" OPID1
	launch_xterm "./empire-captain/ai${IA1}.py localhost ${PPORT1} 2>&1 | tee out_P1" PPID1
	launch_xterm "./empire-tee/tee.py localhost ${SPORT} ${PPORT2} ${OPORT2}" TPID2
	launch_xterm "./empire-client/Main.native -obs -sport ${OPORT2}" OPID2
	launch_xterm "python ./DeepLearningGame_TensorFlow/main.py localhost ${PPORT2} > out_IA | tee out_P2" PPID2

	PIDS="${SPID} ${TPID1} ${OPID1} ${PPID1} ${TPID2} ${OPID2} ${PPID2}"

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
	echo "END PART"
	echo "END PART $NBWIN0 $NBWIN1"
done
echo $NBWIN0 $NBWIN1
