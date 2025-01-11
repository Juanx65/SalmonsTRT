#!/bin/bash

# remeber to do a chmod +x build.sh befor runing with ./build.sh
BATCH_SIZE=-1 # $1 #128

C=3
W=640
H=640

echo $INPUT_SHAPE

# ONNX
#python onnx_transform.py --weights weights/best.pt --input_shape $BATCH_SIZE $C $H $W

#TRT FP32
python build_trt.py --weights weights/best.onnx  --fp32 --input_shape $BATCH_SIZE $C $H $W --engine_name best_fp32.engine

#python build_trt.py --weights="weights/best_fp16.onnx"  --fp16 --input_shape $BATCH_SIZE $C $H $W --engine_name best_fp16.engine

#TRT INT8
#rm -r outputs/cache
#python build_trt.py --weights="weights/best_int8.onnx"  --int8 --input_shape $BATCH_SIZE $C $H $W --engine_name best_int8.engine

#yolo val segment data=datasets/salmons/salmons.yaml model=weights/best_fp32.engine

##obs puedes usar python onnx_layer_removal.py para ver el grafo del onnx