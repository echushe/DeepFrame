#!/usr/bin/env bash
# Author: Cong Hai Nguyen
# Created: 180602

# SYNTAX:
# <prog-name> [dnn or cnn] [test filename to try]


# _DELTA_SAVE= (fixed with default)
# read current progress if any:
# if [ -f "${_SAVED_STATE_FNAME}" ] ; then
# save network file into index to avoid re-use/overwritten
# write current progress to pick up later

# E: epoch size (default 10)
# M: momentum (default 0.3)
# N: type of network used (“dnn”, “cnn”, etc)
# S: epoch counts between two adjacent network saves (default 100)
# a: number of threads (default 4)
# b: batch size (default 8)
# e: number of epochs (default 100)
# f: filename of network state when the program starts (if any)
# h: if this flag is set, the network’s weights will be saved to file (default false)
# l: learning rate (default 0.001)
# n: filename for saving network states
# s: seed (default 1.0)
# t: filename containing a list of training files
# 1: filename containing a list of files for the first test
# 2: filename containing a list of files for the second test

# FIXED:
# epoch between save: fixed with default
# epoch_size: default 10 (fixed)
# learning rate: fixed

set -u


# VARIABLES:

# GO TO SOURCE DIR:
readonly CWD="$(pwd)"
readonly _THIS_REALFILE="$(readlink -e "${BASH_SOURCE}")"
readonly _THIS_REALDIR="$(dirname "${_THIS_REALFILE}")"
readonly _TEST_ROOT_DIR="${_THIS_REALDIR}"
readonly _BIN_DIR="${_THIS_REALDIR}"/../bin/x64/Release/

cd "${_BIN_DIR}" ; _RETVAL=$?

if [ $_RETVAL -ne 0 ] ; then
    echo "cannot get to ${_BIN_DIR} from ${CWD}" >&2
    exit $_RETVAL
fi

# PARSE ARGUMENTS:

start_value=0
end_value=''
data_type=''

while [ $# -gt 0 ] ; do
    case "$1" in
	-d) shift 1
	    set -x
	    ;;
	-0) start_value="$2"
	    shift 2
	    ;;
	-1) end_value="$2"
	    shift 2
	    ;;
	-t)
	    data_type="$2"
	    shift 2
	    ;;
	*)
	    break
	    ;;
    esac
done

if [ -z "$data_type" ] ; then
    echo "ERROR: no data type (epoch, batch size, etc) specified on command line" >&2
    exit 1
fi

all_network_type_ls='dnn cnn'

case "$#" in
    0)
	argv_network_type_ls="${all_network_type_ls}"
    ;;
    1)
	if [ "$1" == "all" ] ; then
	    argv_network_type_ls="${all_network_type_ls}"
	else
	    TEST_OUTPUT_FILEDIR="${_TEST_ROOT_DIR}/$1"
	    if [ -f "${TEST_OUTPUT_FILEDIR}" ] ; then
		facetrain_args="$(head -1 ${TEST_OUTPUT_FILEDIR})"
		./facetrain.out ${facetrain_args}
		exit $?
	    elif (echo "${all_network_type_ls}" | grep -qF "$1" ) ; then
		argv_network_type_ls="$1"
	else
		echo "ERROR: unrecognized command-line arg: '${1}'" >&2
		exit 1
	    fi
	fi
	;;
    *)
	echo "ERROR: unrecognized command-line arg: '${@}'" >&2
	exit 1
	;;
esac

if [ -z "$argv_network_type_ls" ] || ! (echo "${all_network_type_ls}" | grep -qF "${argv_network_type_ls}") ; then
    echo "ERROR: wrong command-line argument: network types: ${argv_network_type_ls}" >&2
    exit 1
fi


case "${data_type}" in
    epoch)
	argv_flag='-b 8 -E 8 -e ' # reduce batch size and epoch size to reduce fast accu of dnn for chart
	argv_value_ls=' 1 2 4 8 16 32 64 128 '
	;;
    momentum)
	argv_flag='-m '
	argv_value_ls=' 0 0.2 0.4 0.8 1.0 '
	;;
    nthreads)
	argv_flag='-a '
	argv_value_ls=' 1 2 4 8 '
	;;
    *)
	echo "ERROR: unrecognized data_type: ${data_type}" >&2
	exit 1
	;;
esac

for network_type in ${argv_network_type_ls} ; do
    TEST_OUTPUT_DIR="${_TEST_ROOT_DIR}/${data_type}_${network_type}"
    mkdir --parents "${TEST_OUTPUT_DIR}"
    echo "using ${network_type} network:"
    # index=0
    base_cmd=''
    for test_category in "shades" "face" "pose" "express" ; do
	for each_value in ${argv_value_ls} ; do
	    if [ "$(expr "$each_value" '<' "$start_value")" == "1" ] ; then
		continue
	    fi
	    if [ -n "$end_value" ] && [ "$(expr "$each_value" '>' "$end_value")" == "1" ] ; then
		break
	    fi
	    test_outfile="${TEST_OUTPUT_DIR}/${test_category}${each_value}"
	    # test_outfile_result="${test_outfile}.result"
	    facetrain_args="-N ${network_type} -n ${test_category} -t all_train.list -1 all_test1.list -2 all_test2.list ${argv_flag}  ${each_value} -S 5000" # -e ${epoch} -a ${nthreads} -b ${batch_size} -m ${momentum} -s ${seed}
	    echo ${facetrain_args} > "${test_outfile}"
	    # start_time="$(date +%s)"
	    # if [ $start_value -gt 1 ] ; then
	    # 	printf "index: ${index}\nargs:\n${facetrain_args}" >&2
	    # fi
	    find -type f -name '[cd]nn_*' -exec rm --verbose '{}' \;
find -type f -regex '\([cd]nn_.*\|.*[.]start\|.*[.]end\)' -exec rm --verbose '{}' \;
	    /usr/bin/time --output "${test_outfile}" --append -f "time_real: %E time_user: %U time_sys: %S" ./facetrain.out ${facetrain_args} | tail -1 >> "${test_outfile}"
	    # end_time="$(date +%s)"
	    if [ $(wc -l "${test_outfile}"  | cut -d' ' -f 1) -lt 2 ] ; then
		printf "\nERROR: no result found after running test with args:\n ${facetrain_args}\n\n" >&2
		exit 1
	    fi
	    # printf "time: " >> "${test_outfile}"
	    # expr "$end_time" '-' "$start_time" >> "${test_outfile}"
	    # printf " start: ${start_time} end: ${end_time}" >> "${test_outfile}"
	done
    done
done
			
    # for epoch in 25 50 75 100 ; do # ($(seq 1 100))
# 	# if [ $epoch -gt 100 ] ; then break ; fi
# 	# epoch=$((epoch_start*epoch_mult))
# 	# epoch_mult=$((epoch_mult*2))
#     done
    
#     for seed in 1.0 ; do # -1.0: (seed will be from system time) - 1.0 (for reproducibility)
# 	# epoch_start=5 # epoch in (seq 5-100) (increment by doubling, save each to log)
# 	# epoch=$epoch_start
# 	# epoch_mult=1
# 	for epoch in 25 50 75 100 ; do # ($(seq 1 100))
# 	    # if [ $epoch -gt 100 ] ; then break ; fi
# 	    # epoch=$((epoch_start*epoch_mult))
# 	    # epoch_mult=$((epoch_mult*2))
# 	    for nthreads in 4 ; do
# 		for batch_size in 4 8 12 16 ; do
# 		    for momentum in 0 0.4 0.8 1.0 ; do
# 			index=$((index+1))
# 			# if [ $(expr "$index" "%" "10") -eq 0 ] ; then
# 			#     echo "processing case ${index} ..."
# 			# fi
	
# 		    done
# 		done
# 	    done
# 	done
#     done
# done
# set +x
