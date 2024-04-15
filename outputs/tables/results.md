# obs

- los tiempos y la estructrura de la tabla se obtienen con `main.sh`
- por ahora solo obtengo los mAP usando `yolo val segment data=datasets/salmons/salmons.yaml model=weights/best_fp16.engine` para cada modelo.
- no puedo obtener las capas ni parametros por la metadata inerente a ultralytics no me permite usar los codigos diseñados para engines sin metadata
- aunque los engines no tubieran metadata, tampoco fui capas de sacar el nimero de parametros, la estructura de la red en modo engine no cuanta con "Weights" que es lo que busca mi codigo para contar los parametros..

# jetson orin agx bs 1

|  Model          | inf/s +-95% | Latency (ms) +-95%|size (MB)  | mAP50 |mAP50-95 | #layers | #parameters|
|-----------------|-------------|-------------------|-----------|-------|------|---------|------------|
| Vanilla         |  8,9  +0,0 -0,0 | 112.8 / 113.5   +0.4 -0.4 |  88.0      | 0.732 | 0.412  | 960     | 45912659   |
| TRT_fp32        |  11,9  +0,2 -0,2 |  83.9 / 85.1    +1.2 -1.2 |  180.3     | 0.721 | 0.403 | 0       | 0          |
| TRT_fp16        |  24,1  +1,3 -1,4 |  41.5 / 49.3    +2.3 -2.2 |  90.5      |  0.723 |0.404 | 0       | 0          |