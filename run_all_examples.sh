#!/usr/bin/env sh

set -e
PREFIX=""

for dir in $(/usr/bin/ls examples); do
    echo -e "\033[33;1m===> running example \"${dir}\"\033[0m"
    ${PREFIX} python3 -m beancode "examples/${dir}"
done
