# Remember to run chmod +x experiments/main.sh before executing ./experiments/main.sh

BATCH_SIZE=$1  # Batch size to use in the experiment, from 1 to 128
NETWORK=$2  # Neural network to use in the experiment
BUILD_TYPE=$4 # "dynamic" or "static" Batch size to build your engines
OP_LVL=$5 # Builder optimization level (int from 0 to 5, default = 3)
DATASET_PATH=$6  # Validation dataset path, e.g., "datasets/dataset_val/val"; if none, the script will execute a test with random inputs
POWER_MODE=$7 #if you profile on a specific power mode, specify it for the name of the logs
PROFILE=$8 # write "pytorch" to profile with with pytorch profiler, "tegrastats" to profile with tegrastats or leave it blank.

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

# EXECUTIONS
# Vanilla (BASE MODEL) WE ADD --engine to indicate the ONNX of origin; with this ONNX, we calculate the network parameters
# If you do not specify a dataset, this program will perform an evaluation with random inputs, providing only latency results.
PROFILER=" "
if [ "$PROFILE" = "pytorch" ]; then
    PROFILER="--profile"
fi

if [ -z "$DATASET_PATH" ] || [ "$DATASET_PATH" == "none" ]; then
    MODEL="experiments/main/main.py --batch_size $BATCH_SIZE --trt --engine weights/${NETWORK}.engine --model_version FP16 --log_dir outputs/log/log_fp16_${NETWORK}_bs_${BATCH_SIZE}_${POWER_MODE} $PROFILER"
else
    MODEL="experiments/main/main.py -v --batch_size $BATCH_SIZE --dataset $DATASET_PATH --network $NETWORK --less --engine weights/best.engine --model_version Vanilla --log_dir outputs/log/log_vanilla_${NETWORK}_bs_${BATCH_SIZE}_${POWER_MODE} $PROFILER"
fi

#sudo rm -r outputs/log > /dev/null 2>&1
#rm post_processing/*.txt > /dev/null 2>&1

# Execute Python scripts sequentially
if [ "$PROFILE" = "tegrastats" ]; then
    execute "$MODEL" "jetson" "outputs/tegrastats_log/vanilla_${NETWORK}_bs_${BATCH_SIZE}_${POWER_MODE}_${BUILD_TYPE}_OPLVL${OP_LVL}.txt"
else
    execute "$MODEL"
fi