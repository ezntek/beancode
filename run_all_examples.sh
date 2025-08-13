#!/usr/bin/env sh

set -e

for dir in $(/usr/bin/ls examples); do
    echo -e "\033[33;1m===> running example \"${dir}\"\033[0m"
    python -m beancode "examples/${dir}"
done
