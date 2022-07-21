#!/usr/bin/bash

files=$(ls $1)

max=0
for file in $files
do
	f_max=$(cat $1/$file | wc -l)
	if [ $f_max -ge $max ]; then
		max=$f_max
	fi
done
echo $max
