#!/bin/sh

file1="${PWD}/output1.txt"
file2="${PWD}/output2.txt"

diff $file1 $file2 > /dev/null 2>&1

if [ $? != 0 ]
then
   echo "Verify failure"
else
   echo "Verify success"
fi

rm $file2
