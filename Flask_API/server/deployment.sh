#!/bin/sh

export TERM=linux
export TERMINFO=/etc/terminfo

script_dir=$(dirname $0)
cd $script_dir; /usr/bin/python3 API_Flask_App.py > API_logger.log 2>&1
