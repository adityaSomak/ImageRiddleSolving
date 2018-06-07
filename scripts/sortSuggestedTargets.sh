#!/bin/sh 

#######################################################################
########		PIPELINE STAGE II(preprocessing)
########		Input: suggested targets
########		Output: sort targets based on vis.similarity+similarity
#######################################################################

for filename in "$1"/*targets_.txt; do
	filenameWithoutExtension="${filename%.*}"
	sort -nr -k3,3 -k2,2 "$filename" > "$filenameWithoutExtension"_sorted.txt;
done
