# Remember to run chmod +x experiments/main.sh before executing ./experiments/main.sh
BATCH_SIZE=$1  # Batch size 1 to 32
WEIGHTS=$2  # WEIGHTS to use in the experiment
modelOP_LVL=$3 # Builder optimization level (int from 0 to 5, default = 3)
POWER_MODE=$4 #if you profile on a specific power mode, specify it for the name of the logs
PROFILE=$5 # write "pytorch" to profile with with pytorch profiler, "tegrastats" to profile with tegrastats or none.
BUILD_TYPE=$6 # -trt or leave blank

C=3  # Number of input channels
W=640  # Input width
H=640  # Input height

# Set the memory threshold (in kilobytes)
MEM_THRESHOLD=102400  # 100MB

# Function to run the Python script
execute() {
    local script=$1
    local is_jetson=$2
    local output_name=$3
    local tegrastat_pid
    local python_pid

    if [ "$is_jetson" = "jetson" ]; then
        tegrastats --interval 1 --logfile $output_name & #sudo tegrastats si necesitas ver mas metricas
        tegrastat_pid=$!
    fi
    # Run the Python script in the background and get its PID
    sudo $script &
    python_pid=$!
    
    #echo "Starting $script with PID $python_pid"

    # Monitor the memory usage of the process
    while true; do
        # Check if the process has finished
        if ! kill -0 $python_pid 2>/dev/null; then
            #echo "$script with PID $python_pid has finished"
            #echo "$script con PID $python_pid terminó"
            #detenemos tegrastats
            if [ "$is_jetson" = "jetson" ]; then
                kill -9 $tegrastat_pid
            fi
            break
        fi

        # Get the memory usage of the process
        mem_avail=$(grep MemAvailable /proc/meminfo | awk '{print $2}') 
        #echo "Available memory: $mem_avail"

        if [ "$mem_avail" -lt "$MEM_THRESHOLD" ]; then
            echo "Memory exceeded ($mem_avail KB available) by $script, terminating PID $python_pid"
            if [ "$is_jetson" = "jetson" ]; then
                kill -9 $tegrastat_pid
            fi
            kill -9 $python_pid
            break
        fi
        sleep .1  # Wait for 0.1s before the next check
    done
}

MODEL="experiments/main/main.py --less --batch_size $BATCH_SIZE --weights $WEIGHTS $BUILD_TYPE"

if [ "$PROFILE" = "tegrastats" ]; then
    execute "$MODEL" "jetson" "outputs/tegrastats_log/${WEIGHTS}_bs_${BATCH_SIZE}_${POWER_MODE}_${BUILD_TYPE}_OPLVL${OP_LVL}.txt"
else
    execute "$MODEL"
fi