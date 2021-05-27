## ML@Edge with SageMaker - Getting Started Kit

| Board           | Model                                                         |Processor    | Cold start (no TRT cache) | Cold start time (TRT cache)     | Inference time  |
| -               | -                                                             | -           | -               | -                   | -                   |
| Jetson Nano     | [Tiny Yolov4 416x416 80 classes](models/02_YoloV4/01_Pytorch) |GPU          | ~98.02s         | -             | ~73ms               |
| Jetson Nano     | [Tiny Yolov4 416x416 80 classes](models/02_YoloV4/01_Pytorch) |CPU          | ~0.86s          | -              | ~1503ms             |
| Jetson Xavier   | [Tiny Yolov4 416x416 80 classes](models/02_YoloV4/01_Pytorch) |GPU          | x               | -                   | x               |
| Jetson Xavier   | [Tiny Yolov4 416x416 80 classes](models/02_YoloV4/01_Pytorch) |CPU          | x               | -                   | x               |

## SageMaker Edge Manager Docker container
[Instructions](sagemaker_edge_manager_agent_docker/README.md) of how to create a docker container for the ARM64 agent + Nvidia devices -> Nano/Xavier.
