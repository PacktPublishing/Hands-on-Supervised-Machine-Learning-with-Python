#!/bin/bash

set -e

run_tests() {
    # run the tests with coverage
    pytest --cov packtml
}

run_tests
