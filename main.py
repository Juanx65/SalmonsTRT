import argparse
import os
import time

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import numpy as np

from utils.data_loader import create_custom_loaders
from utils.helper import AverageMeter

from torch.profiler import profile, ProfilerActivity
#from hta.trace_analysis import TraceAnalysis

#from torchinfo import summary

import subprocess
import re

import scipy.stats as stats
from ultralytics import YOLO

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

best_prec1 = 0.0

def main(opt):
    train_on_gpu = torch.cuda.is_available()
    if not opt.non_verbose:
        if not train_on_gpu:
            print('CUDA is not available.')
        else:
            print('CUDA is available.')

    global best_prec1, device
    device = torch.device("cuda:0" if train_on_gpu else "cpu")

    model = YOLO(opt.weights, task='segment',verbose=False)

    #if not opt.trt:
    #    model = model.model.fuse().to(device)
    
    current_directory = os.path.dirname(os.path.abspath(__file__))
    yaml_path = current_directory+'/'+opt.dataset+'/salmons.yaml'
    # Crear DataLoaders
    _, val_loader = create_custom_loaders(yaml_path, opt.batch_size, opt.workers, opt.pin_memmory, do_normalize=False)

    evaluate(opt,val_loader, model)
    return


def evaluate(opt,val_loader, model):

    nun_batches = 100
    batch_time_all = AverageMeter()
    size_MB = get_model_size_MB(opt)

    # Calculate 10% of total batches
    warmup_batches = int(0.1 * nun_batches)
    
    # Initialize the maximum and minimum processing time after warm-up
    max_time_all = 0
    min_time_all = float('inf')

    num_batches_to_process = int(1 * nun_batches)

    for i, (input, target) in enumerate(val_loader):
        if i >= num_batches_to_process:
            break
        if input.size(0) != opt.batch_size:
            print(f"Deteniendo la evaluación. Tamaño del lote ({input.size(0)}) no es igual a batch_size ({opt.batch_size}).")
            break
        target = target.to(device)

        start_all = time.time() # start time, moving data to gpu
        input = input.to(device)

        with torch.no_grad():
            #output = model(input)
            output = model.predict(input, save=False, imgsz=640, conf=0.25)[0]
            
            if isinstance(output, tuple):
                output_cpu = tuple(o.cpu() if torch.is_tensor(o) else o for o in output)
            elif isinstance(output, list):
                output_cpu = list(o.cpu() if torch.is_tensor(o) else o for o in output)
            else:
                output_cpu = output.cpu()
            #output_cpu = output.cpu() # con proposito de calcular el tiempo que tarda en volver a pasar la data a la cpu
            all_time = (time.time() - start_all) * 1000  # Convert to milliseconds / time when the result pass to cpu again 

        # measure elapsed time in milliseconds and ignore first 10% batches
        if i >= warmup_batches:
            
            batch_time_all.update(all_time)
            max_time_all = max(max_time_all, all_time)
            min_time_all = min(min_time_all, all_time)
    
    #----------------------------------------------------------------------------------------------------------#
    #                           INTERVALO DE CONFIANZA 95%                                                     #
    #----------------------------------------------------------------------------------------------------------#
    # Supongamos que tienes tus datos de tiempos en una lista llamada data
    data = batch_time_all.values  # Esta es la lista de tus tiempos
    # Estimación de parámetros de la distribución gamma
    alpha_hat, loc_hat, beta_hat = stats.gamma.fit(data, floc=0)  # Forzamos a que la localización (loc) sea 0
    mean_gamma = alpha_hat * beta_hat
    # Calcular el intervalo de confianza del 95%
    ci_lower = stats.gamma.ppf(0.025, alpha_hat, scale=beta_hat)
    ci_upper = stats.gamma.ppf(0.975, alpha_hat, scale=beta_hat)

    # Margen de error inferior y superior
    margin_error_lower = mean_gamma - ci_lower
    margin_error_upper = ci_upper - mean_gamma

    infxs = (opt.batch_size*1000 ) / batch_time_all.avg # inferenicas por segundo
    infxs_me_up = (opt.batch_size*1000 ) / batch_time_all.avg - (opt.batch_size*1000 ) / (batch_time_all.avg + margin_error_upper) # marginal error inferencias por segundo intervalo de confianza 95%
    infxs_me_low = (opt.batch_size*1000 ) / (batch_time_all.avg - margin_error_lower) -  (opt.batch_size*1000 ) / batch_time_all.avg # marginal error inferencias por segundo intervalo de confianza 95%
    
    #----------------------------------------------------------------------------------------------------------#
    #                                                                                                          #
    #----------------------------------------------------------------------------------------------------------#
    
    #n = len(val_loader) - warmup_batches
    #print("n: ", n)
    #print("pasa marginal: ", pasa_marginal) # asegurarse de que sea menor del 5% para que el calculo sea correcto
    if opt.trt:
        total_parametros = get_parametros(opt)
        total_capas = get_layers(opt)
    else:
        total_capas, total_parametros = get_parameters_vanilla(opt,model)

    metrics = model.val(data='datasets/salmons/salmons.yaml', task='segment', batch=opt.batch_size, verbose=False)
        
    if not opt.non_verbose:
        print("|  Model          | inf/s +-95% | Latency (ms) +-95%|size (MB)  | mAP50 |mAP50-95 | #layers | #parameters|")
        print("|-----------------|-------------|-------------------|-----------|-------|------|---------|------------|")
    print("| {:<15} |  {:}  +{:} -{:} | {:>5.1f} / {:<6.1f}  +{:.1f} -{:.1f} |  {:<9.1f} | {:<20.2f} | {:<19.2f} | {:<6}  | {:<9}  |".format(
        opt.model_version, 
        number_formater(infxs) ,number_formater(infxs_me_up) ,number_formater(infxs_me_low),
        batch_time_all.avg, max_time_all, margin_error_upper,margin_error_lower,
        size_MB, 
        metrics.box.map50, metrics.box.map,
        total_capas,total_parametros))

    if opt.histograma:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.hist(batch_time_all.values, bins=50, color='blue', alpha=0.7)
        plt.title('Distribución de batch_time_all')
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        plt.savefig("diosmelibre.pdf", bbox_inches='tight',format='pdf')

    return

# eval with random inputs 
def evaluate_random(opt, model):
    nun_batches = 100
    torch.manual_seed(42)
    inputs= torch.rand(nun_batches,opt.batch_size, 3, 640, 640) # generamos un input random [0,1)
    #summary(model,input_size=(opt.batch_size, 3,640,640)) # no funciona con engine.

    batch_time_all = AverageMeter()

    # switch to evaluate mode
    #model.eval()
    # Supongamos que 'model' es tu modelo PyTorch
    size_MB = get_model_size_MB(opt)

    # Calculate 10% of total batches
    warmup_batches = int(0.1 * nun_batches)
    
    # Initialize the maximum and minimum processing time after warm-up
    max_time_all = 0
    min_time_all = float('inf')

    num_batches_to_process = int(1 * nun_batches)

    for i in range(nun_batches):
        input = inputs[i].to(device)
        if i >= num_batches_to_process:
            break
        # Comprobar el tamaño del lote
        if input.size(0) != opt.batch_size:
            if not opt.less:
                print(f"Deteniendo la evaluación. Tamaño del lote ({input.size(0)}) no es igual a batch_size ({opt.batch_size}).")
            break

        start_all = time.time() # start time, moving data to gpu
        input = input.to(device)
        
        with torch.no_grad():
            if opt.trt:
                output = model(input, verbose=False)
            else:
                output = model(input)
            
            if isinstance(output, tuple):
                output_cpu = tuple(o.cpu() if torch.is_tensor(o) else o for o in output)
            elif isinstance(output, list):
                output_cpu = list(o.cpu() if torch.is_tensor(o) else o for o in output)
            else:
                output_cpu = output.cpu()
            #output_cpu = output.cpu() # con proposito de calcular el tiempo que tarda en volver a pasar la data a la cpu
            all_time = (time.time() - start_all) * 1000  # Convert to milliseconds / time when the result pass to cpu again 

        # measure elapsed time in milliseconds and ignore first 10% batches
        if i >= warmup_batches:
            
            batch_time_all.update(all_time)
            max_time_all = max(max_time_all, all_time)
            min_time_all = min(min_time_all, all_time)
    
    #----------------------------------------------------------------------------------------------------------#
    #                           INTERVALO DE CONFIANZA 95%                                                     #
    #----------------------------------------------------------------------------------------------------------#
    # Supongamos que tienes tus datos de tiempos en una lista llamada data
    data = batch_time_all.values  # Esta es la lista de tus tiempos
    # Estimación de parámetros de la distribución gamma
    alpha_hat, loc_hat, beta_hat = stats.gamma.fit(data, floc=0)  # Forzamos a que la localización (loc) sea 0
    mean_gamma = alpha_hat * beta_hat
    # Calcular el intervalo de confianza del 95%
    ci_lower = stats.gamma.ppf(0.025, alpha_hat, scale=beta_hat)
    ci_upper = stats.gamma.ppf(0.975, alpha_hat, scale=beta_hat)

    # Margen de error inferior y superior
    margin_error_lower = mean_gamma - ci_lower
    margin_error_upper = ci_upper - mean_gamma

    infxs = (opt.batch_size*1000 ) / batch_time_all.avg # inferenicas por segundo
    infxs_me_up = (opt.batch_size*1000 ) / batch_time_all.avg - (opt.batch_size*1000 ) / (batch_time_all.avg + margin_error_upper) # marginal error inferencias por segundo intervalo de confianza 95%
    infxs_me_low = (opt.batch_size*1000 ) / (batch_time_all.avg - margin_error_lower) -  (opt.batch_size*1000 ) / batch_time_all.avg # marginal error inferencias por segundo intervalo de confianza 95%
    
    #----------------------------------------------------------------------------------------------------------#
    #                                                                                                          #
    #----------------------------------------------------------------------------------------------------------#
    
    #n = len(val_loader) - warmup_batches
    #print("n: ", n)
    #print("pasa marginal: ", pasa_marginal) # asegurarse de que sea menor del 5% para que el calculo sea correcto
    if opt.trt:
        total_parametros = get_parametros(opt)
        total_capas = get_layers(opt)
    else:
        total_capas, total_parametros = get_parameters_vanilla(opt,model)
        
    if not opt.non_verbose:
        print("|  Model          | inf/s +-95% | Latency (ms) +-95%|size (MB)  | mAP@1 |mAP@5 | #layers | #parameters|")
        print("|-----------------|-------------|-------------------|-----------|-------|------|---------|------------|")
    print("| {:<15} |  {:}  +{:} -{:} | {:>5.1f} / {:<6.1f}  +{:.1f} -{:.1f} |  {:<9.1f} | {:<20.2f} | {:<19.2f} | {:<6}  | {:<9}  |".format(
        opt.model_version, 
        number_formater(infxs) ,number_formater(infxs_me_up) ,number_formater(infxs_me_low),
        batch_time_all.avg, max_time_all, margin_error_upper,margin_error_lower,
        size_MB, 
        0.0, 0.0,
        total_capas,total_parametros))

    if opt.histograma:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.hist(batch_time_all.values, bins=50, color='blue', alpha=0.7)
        plt.title('Distribución de batch_time_all')
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        plt.savefig("diosmelibre.pdf", bbox_inches='tight',format='pdf')

    return

def number_formater(numero):
    return "{:,.1f}".format(numero).replace(",", "X").replace(".", ",").replace("X", ".")

def validate_profile(val_loader, model):
    
    num_batches_to_process = int(1/5 * len(val_loader))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        #record_shapes=True,
        #with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/log_trt')) as prof:

        #elapsed_time = 0
        for i, (input, target) in enumerate(val_loader):
            if i >= num_batches_to_process:
                break
            start= time.time()
            input = input.to(device)
            with torch.no_grad():
                output = model(input)
                output = output.cpu()
                #elapsed_time += (time.time() - start) * 1000 

            prof.step() 
    
    #print("avg time: ", elapsed_time/100, " ms")
    return

def get_model_size_MB(opt):
    return os.path.getsize(opt.weights) / (1024 * 1024) 

def get_parameters_vanilla(opt, model):
    total_capas = sum(1 for _ in model.modules())
    total_parametros = sum(p.numel() for p in model.parameters())
    #summary(model, (3,224,224)) ## summary modelo pth o pt segun pytorch
    return total_capas, total_parametros

def get_layers(opt):
    # para que funcione como sudo es necesario correr desde el path del enviroment env/bin/polygraphy
    if opt.trt:
        cmd = f"env/bin/polygraphy inspect model {opt.weights}"
    else:
        cmd = f"env/bin/polygraphy inspect model {(opt.weights).replace('.pt', '.onnx')} --display-as=trt"

    # Ejecuta el comando y captura la salida
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Decodifica la salida a texto
    output = stdout.decode()

    # Usa una expresión regular para encontrar el número de capas
    match = re.search(r"---- (\d+) Layer\(s\) ----", output)
    # Extrae el número de capas si se encuentra el patrón
    if match:
        num_layers = int(match.group(1))
        return num_layers
    else:
        print("No se encontró el número de capas")
        return 0

def get_parametros(opt):
    if opt.trt:
        cmd = f"env/bin/python utils/param_counter.py --engine ../{opt.weights}"
    else:
        cmd = f"env/bin/onnx_opcounter {(opt.weights).replace('.pt', '.onnx')}"

    # Ejecuta el comando y captura la salida
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Decodifica la salida a texto
    output = stdout.decode()

    # Usa una expresión regular para encontrar el número de capas
    match = re.search(r"Number of parameters in the model: (\d+)", output)
    if match:
        num_parameters = int(match.group(1))
        return num_parameters
    else:
        print("No se encontró el número de parametros")
        return 0

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='datasets/salmons', help='path to dataset')
    parser.add_argument('--batch_size', default = 1, type=int,help='batch size to train')
    parser.add_argument('--weights', default = 'weights/yolov8lsalmons.pt', type=str, help='directorio y nombre de archivo de donse se guardara el mejor peso entrenado')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')
    parser.add_argument('-m','--pin_memmory', action='store_true',help='use pin memmory')
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',help='print frequency (default: 10)')
    parser.add_argument('-trt','--trt', action='store_true',help='evaluate model on validation set al optimizar con tensorrt')
    parser.add_argument('-n','--network', default='resnet18',help='name of the pretrained model to use')
    parser.add_argument('-v','--validate', action='store_true',help='validate with validation data')
    parser.add_argument('-c','--compare', action='store_true',help='compare the results of the vanilla model with the trt model using random generated inputs')
    parser.add_argument('-rtol','--rtol', default=1e-3,type=float, help='relative tolerance for the numpy.isclose() function')
    parser.add_argument('-vd','--val_dataset', action='store_true',help='compare the results of the vanilla model with the trt model using the validation dataset as inputs')
    parser.add_argument('--profile', action='store_true',help='profiles the validation run with torch profiler')
    parser.add_argument('--compare_3', action='store_true',help='compare the results of the vanilla model with the trt model using random generated inputs')
    parser.add_argument('--less', action='store_true',help='print less information')
    parser.add_argument('--non_verbose', action='store_true',help='no table header and no gpu information')
    parser.add_argument('--model_version', default='Vanilla',help='model name in the table output (validation): Vanilla, TRT fp32, TRT fp16 TRT int8')
    parser.add_argument('--histograma', action='store_true',help='guarda una figura con el histograma de los tiempos x inferencia de cada batch')#'./log/log_vnll'}
    parser.add_argument('--log_dir', default='log/log_vnll', help='path to log dir for pytorch profiler')
   
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)