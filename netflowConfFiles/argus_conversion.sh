#!/bin/bash

rarcPath="${1}/rarc"
filesPath="${2}"



generateOutputArgusFiles () {
    for filename in $filesPath/*.argus; do
	echo $filename
    	ra -F $rarcPath -r $filename > ${filename/argus/txt}
    done
}

generateOutputArgusFiles


