#!/bin/bash

# test a single net for the tegrastat metrics
tegrastats --interval 1 --logfile tegrastats_test.txt & #sudo tegrastats si necesitas ver mas metricas
tegrastat_pid=$!
python experiments/main/main.py --less --batch_size 8 --weights weights/yolov8lsalmons_fp16_bs32.engine --trt
kill -9 $tegrastat_pid