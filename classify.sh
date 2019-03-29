#!/bin/bash

tmp_file=$(mktemp)
key=^$2 
cat $1|grep $key>tmp_file 

./knn_cross_validation.py tmp_file ../kimia99/
