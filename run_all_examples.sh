#!/usr/bin/env sh

set -e

for f in examples/*; do
    echo -e "\033[33;1m===> running example \"$f\"\033[0m"
    ${PYTHON:-python3} -m beancode $@ "$f" 
done
