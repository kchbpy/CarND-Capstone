# Udacity CarND Capstone project

Udacity Self Driving Car final project.
About the project overview, please refer to [README.org.md](./README.org.md).

## Capstone project team member

| Name                  | Email                      |
|-----------------------|----------------------------|
| Satoshi Kumano (Lead) | satoshi.kumano@gmail.com   |
| Nghi Tran             | nathantran99@gmail.com     |
| Wang Xi               | kchbpy@hotmail.com         |
| Naushad Rahman        | naushad.rahman@hotmail.com |
| Chao Liu              | billliu_666@163.com        |

## Traffic Light detection

We used [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

### Pre-trained model:

We used [pre-trained model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) in the repository.

Choose [ssd_mobilenet_v1_coco_2017_11_17](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz) for the first step but have not tested other models yet.

### Training

Details of training with traffic light data in simulator is described in the [repo](https://github.com/satoshikumano/traffic-light-detection-fh).

### TODOs

- Accuracy is not enough.
  - Increase training data/ validation data.
  - Data augmentation.

- Reduce latency of detection.

