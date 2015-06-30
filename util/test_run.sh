#!/bin/bash

echo "--------------------------------------------------------------"

echo "N=$1"

WD=$(pwd)

./dlpoly_argon_generator.py -N $1

DLPOLY.Z

./dlpoly_get_cputime.py



cd ../src
PPMD=$(./argon_example.py -N $1 | grep "integrate time taken")

echo "PPMD $PPMD"

cd $WD











