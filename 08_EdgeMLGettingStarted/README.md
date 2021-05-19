## ML@Edge with SageMaker - Getting Started Kit

| Board       |Processor| Model     | Cold start time     | Inference time |
| :------------- |:----------:| ----------: | -----------: |-----------: |
|  Jetson Nano |GPU| [Tiny Yolov4 416x416 80 classes](models/02_YoloV4/01_Pytorch) | ~98.02s    | ~73ms |
|  Jetson Nano |CPU| [Tiny Yolov4 416x416 80 classes](models/02_YoloV4/01_Pytorch) | ~0.86s    | ~1503ms |
| Jetson Xavier   || [Tiny Yolov4 416x416 80 classes](models/02_YoloV4/01_Pytorch) | x | x |

## SageMaker Edge Manager Docker container
[Instructions](sagemaker_edge_manager_agent_docker/README.me) of how to create a docker container for the ARM64 agent + Nvidia devices -> Nano/Xavier.
