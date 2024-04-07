import torch
import onnx
import os
from io import BytesIO
from ultralytics.nn.autobackend import AutoBackend
from ultralytics import YOLO
from copy import deepcopy

import argparse

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available.')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

def main(opt):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    weights_path = opt.weights
    weights_path = os.path.join(current_directory,weights_path)

    YOLOv8 = AutoBackend(opt.weights)
    model = YOLOv8.model.fuse().to(device)
    for m in model.modules():
        # includes all Detect subclasses like Segment
        m.dynamic = True if opt.input_shape[0] == -1 else False
        m.export = True
        m.format = 'onnx'

    save_path = weights_path.replace('.pt', '.onnx')
    if opt.input_shape[0] == -1:
        fake_input = torch.zeros(16,opt.input_shape[1], opt.input_shape[2],opt.input_shape[3]).to(device)
    else:
        fake_input = torch.zeros(1,opt.input_shape[1], opt.input_shape[2],opt.input_shape[3]).to(device)
    
    y = None
    for _ in range(2):
        y = model(fake_input)  # dry runs
    
    print("model stride: ", int(max(model.stride)))
    
    dynamic = {}
    if opt.input_shape[0] == -1:
        dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
        dynamic["output0"] = {0: "batch", 2: "anchors"} # shape(1, 116, 8400)
        dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"} # shape(1,32,160,160)
    print("dynamic: ", dynamic)
    print("fake input: ", fake_input.shape)
    torch.onnx.export(
            model,
            fake_input,
            f=save_path,
            verbose=False,
            opset_version=get_latest_opset(),
            do_constant_folding=False,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
            input_names=["images"],
            output_names=["output0", "output1"],
            dynamic_axes=dynamic if opt.input_shape[0] == -1 else None,
        )
    
    onnx_model = onnx.load(save_path)  
    onnx.checker.check_model(onnx_model) 

    # Guardar el modelo ONNX en un archivo .onnx
    onnx.save(onnx_model, save_path)

    print("La conversión a ONNX se ha completado exitosamente. El modelo se ha guardado en:", save_path)

def get_latest_opset():
    """Return second-most (for maturity) recently supported ONNX opset by this version of torch."""
    return max(int(k[14:]) for k in vars(torch.onnx) if "symbolic_opset" in k) - 1  # opset

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default = 'weights/best.pth', type=str, help='path to the pth weight file')
    parser.add_argument('-p','--pretrained', action='store_true',help='transform a pretrained model from torch.hub.load')
    parser.add_argument('-n','--network', default='resnet18',help='name of the pretrained model to use')
    parser.add_argument('--input_shape',
                        nargs='+',
                        type=int,
                        default=[-1,3, 640,640],
                        help='Model input shape, el primer valor es el batch_size, -1 (dinamico))]')
    
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)