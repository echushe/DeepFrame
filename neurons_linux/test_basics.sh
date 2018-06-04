#!/usr/bin/env bash

readonly CWD="$(pwd)"
readonly _THIS_REALFILE="$(readlink -e "${BASH_SOURCE}")"
readonly _THIS_REALDIR="$(dirname "${_THIS_REALFILE}")"
readonly _TEST_ROOT_DIR="${_THIS_REALDIR}"
readonly _BIN_DIR="${_THIS_REALDIR}"/facetrain/bin/x64/Release/

cd "${_BIN_DIR}" ; _RETVAL=$?

if [ $_RETVAL -ne 0 ] ; then
    echo "cannot get to ${_BIN_DIR} from ${CWD}" >&2
    exit $_RETVAL
fi

echo "recognizing sunglasses (epoch 75):"

./facetrain.out -n shades -t straightrnd_train.list -1 straightrnd_test1.list -2 straightrnd_test2.list -e 75

echo
echo
echo "recognizing faces (epoch 100):"

./facetrain.out -n face -t straightrnd_train.list -1 straightrnd_test1.list -2 straightrnd_test2.list -e 100

echo
echo
echo "recognizing faces (different inputs):"
./facetrain.out -n face -T -t straightrnd_train.list -1 straighteven_test1.list -2 straighteven_test2.list

echo
echo
echo "recognizing poses:"
./facetrain.out -n pose -t all_train.list -1 all_test1.list -2 all_test2.list -e 100

echo
echo
echo "get weights of hidden layers and save them as images:"

./facetrain.out -n shades -t straightrnd_train.list -1 straightrnd_test1.list -H
./facetrain.out -n face -t straightrnd_train.list -1 straightrnd_test1.list -H
./facetrain.out -n pose -t straightrnd_train.list -1 straightrnd_test1.list -H


