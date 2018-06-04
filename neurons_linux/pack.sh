#!/usr/bin/env bash
# Author: Cong Hai Nguyen
# Created: 180601

_THIS_REALFILE="$(readlink -e "${0}")"
_THIS_REALDIR="$(dirname "${_THIS_REALFILE}")"

cd "${_THIS_REALDIR}"

_VERBOSE=--verbose
_CWD="$(pwd)"
# _REGEX='.*[.]\(h\|cpp\|sh\|sln\|py\)'
# _NOT_REGEX='.*[.]\(out\|\)'
# _NOT_DIR_REGEX='.*\(cmu_facial\|[.]git\|[.]vs\|obj\|x64\|Debug\|Release\|bin\).*'
_NOT_DIR_REGEX=' .git .vs obj x64 Debug Release bin'  # cmu_facial
_YES_FILE='facetrain.out'
_FROM_ROOT_DIR=.
_BASENAME="$(basename "$(readlink -f "${_FROM_ROOT_DIR}")")"
_TO_ROOT_DIR=../files/"${_BASENAME}" #  # MUST BE OUTSIDE CURRENT DIR - OTHERWISE WILL CAUSE INFINITE LOOP WHEN -exec in 'find' command
_BASENAME_TAR=files.tgz
_TO_ROOT_DIR_BASENAME="$(basename "${_TO_ROOT_DIR}")"
_PACKME_FILES="../packme-files.txt" # DONT put in current dir - may cause infinite loop
# _UNPACK_SCRIPT="$0"


set -ex

rm "${_VERBOSE}" -rf "${_BASENAME_TAR}" "${_TO_ROOT_DIR}" "${_PACKME_FILES}"

mkdir --parents --verbose "${_TO_ROOT_DIR}"

# cp --force "${_VERBOSE}" "${_TO_ROOT_DIR}"/..

_RSYNC_EXLUDE="$(echo "${_NOT_DIR_REGEX}" | sed --expression 's/ / --exclude=/g')"

# find "${_FROM_ROOT_DIR}" -type d ! -regex "${_NOT_DIR_REGEX}" -exec mkdir --parents "${_VERBOSE}" "${_TO_ROOT_DIR}"/'{}' \;

# find "${_FROM_ROOT_DIR}" -type f ! -regex "${_NOT_DIR_REGEX}" -exec cp "${_VERBOSE}" '{}' "${_TO_ROOT_DIR}"/'{}' \; | tee a- "${_PACKME_FILES}"

rsync --verbose -Rr ${_RSYNC_EXLUDE} "${_FROM_ROOT_DIR}" facetrain/bin/x64/Debug/facetrain.out "${_TO_ROOT_DIR}" | sed --expression '/^.*\/$/d;/^sending /d;/^created /d;/^sent /d;/^total size /d;/^[[:space:]]*$/d' | tee -a "${_PACKME_FILES}"

cd "${_TO_ROOT_DIR}"/..

tar -czf "${_BASENAME_TAR}" "${_TO_ROOT_DIR_BASENAME}"

rm -rf "${_TO_ROOT_DIR}"

mv "${_VERBOSE}" "${_BASENAME_TAR}" "${_CWD}"

cd "${_CWD}"

set +ex

echo "POST VALIDATION:"
printf "File-count from source:"
wc -l "${_PACKME_FILES}"
rm "${_VERBOSE}" "${_PACKME_FILES}"

printf "File-count from dest:"
tar -tf "${_BASENAME_TAR}" | sed --expression '/^.*\/$/d' | wc -l

printf "Compressed file-size: "
du -sh "${_BASENAME_TAR}"

cd "${_CWD}"

# Local Variables:
# firestarter: (shell-command "copy install.sh ..\\..\\")
# End:
