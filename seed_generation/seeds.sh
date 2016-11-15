#!/bin/sh 
 # Shell script to find out all the files under a directory and 
 #its subdirectories. This also takes into consideration those files
 #or directories which have spaces or newlines in their names 

while IFS='' read -r line || [[ -n "$line" ]]; do
    cat "$line" | tr "{" "\n" | tr "]" "\n" | grep "classes" | sed -e "s/[\[,]/\t/g" | sed -e "s/u'//g" >> op.txt;
done < "$1"
#cat $1 | tr "{" "\n" | tr "]" "\n" | grep "classes" | sed -e "s/[\[,]/\t/g" | sed -e "s/u'//g" >> op.txt ;
