#!/bin/bash

cat stl2_v00.txt | xargs -n 2 -P 8 sh -c 'wget --no-check-certificate $0 -O ../stl_files/$1'