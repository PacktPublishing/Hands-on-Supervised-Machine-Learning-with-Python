#!/bin/bash

set -e
codecov || echo "Coverage upload failed"
