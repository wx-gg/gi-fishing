#!/bin/bash
# 
ffmpeg -r 1 -i $1 -r 1 -start_number 0 "frame%04d.png"