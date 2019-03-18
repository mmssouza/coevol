#!/bin/bash
tmp_file=$(mktemp)
awk '/^359/' $1|cut -d\  -f5->$tmp_file
cat $tmp_file
$HOME/coevol/Scatter2.py $tmp_file $HOME/kimia99/
