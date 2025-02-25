import argparse
from utils.engine import EngineBuilder

BATCH_SIZE = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        type=str,
                        default='weights/best.onnx',
                        help='Weights file')
    parser.add_argument('--input_shape',
                        nargs='+',
                        type=int,
                        default=[BATCH_SIZE,3, 640,640],
                        help='Model input shape, el primer valor es el batch_size, 128)]')
    parser.add_argument('--min_hw',
                        type= int,
                        default=320,
                        help='Minimum height and width value, let on cero')
    parser.add_argument('--max_hw',
                        type= int,
                        default=960,
                        help='Max height and width value, let on the doble of the optimus value you wanna get')
    parser.add_argument('--max_batch_size',
                        type= int,
                        default=16,
                        help='max batch size when dynamic batch size')
    parser.add_argument('--dynamic_hw',
                        action='store_true',
                        help='dynamic hw')
    parser.add_argument('--fp32',
                        action='store_true',
                        help='Build model with fp32 mode')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='Build model with fp16 mode')
    parser.add_argument('--int8',
                        action='store_true',
                        help='Build model with int8 mode')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT builder device')
    parser.add_argument('--seg',
                        action='store_true',
                        help='Build seg model by onnx')
    parser.add_argument('--build_op_lvl',
                        default=3,
                        type=int,
                        help='builder optimization level, default 3')
    parser.add_argument('--avg_timing_iterations',
                        default=1,
                        type=int,
                        help='iterations for avg timing, default 1')
    parser.add_argument('--engine_name',
                        default='best.engine',
                        help='name of the engine generated')
    args = parser.parse_args()
    #assert len(args.input_shape) == 4
    return args


def main(args):
    builder = EngineBuilder(args.weights, args.device)
    builder.seg = args.seg
    builder.build(fp32=args.fp32, fp16=args.fp16, int8=args.int8, input_shape=args.input_shape,dynamic_hw=args.dynamic_hw,min_hw=(args.min_hw,args.min_hw),max_hw=(args.max_hw,args.max_hw),max_batch_size=args.max_batch_size,build_op_lvl=args.build_op_lvl,avg_timing_iterations= args.avg_timing_iterations, engine_name=args.engine_name)

if __name__ == '__main__':
    args = parse_args()
    main(args)