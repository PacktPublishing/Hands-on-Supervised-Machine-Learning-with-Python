#!/bin/bash

set -e

export DISPLAY=:99.0
sh -e /etc/init.d/xvfb start
sleep 5 # give xvfb some time to start by sleeping for 5 seconds