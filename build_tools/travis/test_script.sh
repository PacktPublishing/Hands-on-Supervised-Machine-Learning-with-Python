#!/bin/bash

set -e

run_tests() {
    oldpwd=`pwd`

    # Move to another directory to test
    cd ..
    mkdir -p ${TEST_DIR} && cd ${TEST_DIR}
    pytest --cov packtml

    # move back to original dir
    cd ${oldpwd}
}

run_tests
