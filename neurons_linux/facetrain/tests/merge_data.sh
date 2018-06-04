#!/usr/bin/env bash
# Author: Cong Hai Nguyen
# Created: 180602

for category in momentum epoch ; do
    for network_type in dnn cnn ; do
	input_dir="${category}_${network_type}"
	outfile_total="${input_dir}"_raw.txt
	outfile_header="${input_dir}"_header.txt
	outfile_data="${input_dir}"_data.txt
	outfile_final="${input_dir}"_final.txt
	cat "${input_dir}"/* | sed --expression 's/Test accuracy is:/accuracy:/g' | tr -s '\r\n' ' ' | sed --expression 's/\(-N \)/\n\1/g' > "${outfile_total}"
	awk -F ' ' '{ print $1" "$3" "$5" "$7" "$9" "$11" "$13" "$15" "$17" "$19" "$21" "$23 }' "${outfile_total}" > "${outfile_header}"
	awk -F ' ' '{ print $2" "$4" "$6" "$8" "$10" "$12" "$14" "$16" "$18" "$20" "$22" "$24 }' "${outfile_total}" > "${outfile_data}"
	tail -1 "${outfile_header}" > "${outfile_final}"
	cat "${outfile_data}" >> "${outfile_final}"
    done
done

echo "data merged"
