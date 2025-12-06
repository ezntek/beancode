#!/usr/bin/env sh

set -e

${PYTHON:-python3} -m beancode "examples/${1}.bean"
