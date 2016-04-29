#!/usr/bin/bash

TERMFS=10

SPORT=$(( ( RANDOM % 1000 ) + 2000 ))
PPORT1=$(( ( RANDOM % 1000 ) + 3000 ))
PPORT2=$(( ( RANDOM % 1000 ) + 4000 ))
OPORT1=$(( ( RANDOM % 1000 ) + 5000 ))
OPORT2=$(( ( RANDOM % 1000 ) + 6000 ))

IA1=$1

if test x$IA1 = x; then
	echo "usage: $0 <ia1> "
	exit 1
fi

source ./execute-lib.sh ; cd ..
make_empire

# Demarrage des programmes.
launch_xterm "./empire-server/Main.native -sport ${SPORT} 2>&1 | tee out_S" SPID
launch_xterm "./empire-tee/tee.py localhost ${SPORT} ${PPORT1} ${OPORT1}" TPID1
#launch_xterm "./empire-tee/tee.py localhost ${SPORT} ${PPORT1} ${OPORT1} 2>&1 | tee out_T1" TPID1
launch_xterm "./empire-client/Main.native -obs -sport ${OPORT1}" OPID1
launch_xterm "./empire-captain/ai${IA1}.py localhost ${PPORT1} 2>&1 | tee out_P1" PPID1
launch_xterm "./empire-tee/tee.py localhost ${SPORT} ${PPORT2} ${OPORT2}" TPID2
#launch_xterm "./empire-tee/tee.py localhost ${SPORT} ${PPORT2} ${OPORT2} 2>&1 | tee out_T2" TPID2
launch_xterm "./empire-client/Main.native -obs -sport ${OPORT2}" OPID2
launch_xterm "python ./DeepLearningGame_TensorFlow/main.py localhost ${PPORT2} 2>&1 | tee out_P2" PPID2
#launch_xterm "./DeepLearningGame_TensorFlow/main.py localhost ${PPORT2} | tee out_P2" PPID2

#PIDS="${SPID} ${TPID1} ${OPID1} ${PPID1} ${TPID2} ${OPID2} ${PPID2}"
PIDS="${SPID} ${TPID1} ${OPID1} ${PPID1} ${TPID2} ${OPID2}"

# Regarde si un des programmes est stoppe.
STOPPED=0
while test $STOPPED -eq 0; do
	sleep 2
	check_processes STOPPED ${PIDS}
done

# Arret de tous les programmes.
stop_processes ${PIDS}
