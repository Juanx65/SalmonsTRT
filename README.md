# INSTALLATION

## setup jetson nano 

* jetpack 6.0 para orin https://developer.nvidia.com/sdk-manager y seguir instrucciones para setear la nano

## instalar env

* instalar pip y virtualenv

```
sudo apt-get install python3-pip
pip install virtualenv
```

posiblemente salga un mensaje tipo `WARNING: The script virtualenv is installed in '/home/your_user_name/.local/bin' which is not on PATH.

por lo cual es neceasrio añadir al PATH.

```
sudo apt-get install nano
sudo nano ~/.bashrc
```
al final de ~/.bashrc añadir `export PATH=/home/your_user_name/.local/bin:$PATH`

recordar hacer `source ~/.bashrc` para que los cambios se hagan efectivos.

* crear e iniciar virtualenv

```
virtualenv env --system-site-packages
source env/bin/activate
```
* instalar pytorch

Dependiendo de la version de jetpack usada, y segun la pagina ofical de nvnida: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

descargamos el wheel:

`wget https://developer.download.nvidia.cn/compute/redist/jp/v60dp/pytorch/torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl`


instalamos el wheel:

```
sudo apt-get install libopenblas-base libopenmpi-dev  
pip install torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl
```

verificamos;

```
python
>>import torch
>>torch.cuda.is_available()
True
>>exit()
```

* instalar torchvision

descargamos el git de la version el tag segun la version de torhc que instalamos, en mi caso 2.2.0, el tag corresponde segun esto https://github.com/pytorch/vision/releases/tag/v0.17.1
a la v0.17.1

entonces se instala de la siguiente forma:

```
git clone --branch v0.17.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.17.1
python setup.py install
pip3 install pillow==8.4.0
```

* instalar los siguientes pip en este orden: 

`pip install onnx`


`pip install ultralytics`

`pip install numpy --upgrade`

`pip install --upgrade scipy`

* otros pip install:

```
polygraphy
onnx_opcounter
torchinfo
```

---

OBS:

- cambiar el config de ultralitics porque se webea el path al dataset!

- puedes usar `netron` para ver el grafo del onnx!!!

- si no se puede generar int8 intentar remover capas con onnx_layer_removal.py!!
