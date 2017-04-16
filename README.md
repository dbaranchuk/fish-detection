# Fish Detection NCFM

Solution is based on Faster-RCNN using ResNet-50

### Installation

Follow Requirements and Basic Installation [here](https://github.com/rbgirshick/py-faster-rcnn)

### Data

Download the annotated data [here](https://drive.google.com/drive/folders/0BxZ-INMQnIZLdm1CYzlUR1NsOHM?usp=sharing) to the project directory

Download the ResNet-50 pretrained model [here](https://drive.google.com/open?id=0BxZ-INMQnIZLalZyeU00TEtIZzA) to data/imagenet_models/

### Training 

```
./tools/train_net.py --weights data/imagenet_models/ResNet-50-model.caffemodel --imdb fish_train --cfg experiments/cfgs/faster_rcnn_end2end.yml --solver models/fish/ResNet/faster_rcnn_end2end/solver.prototxt
```

### Testing

```
./tools/test_net.py --def models/fish/ResNet/faster_rcnn_end2end/test.prototxt   --net output/faster_rcnn_end2end/train/ResNet_50_faster_rcnn_iter_45000.caffemodel --imdb fish_test --cfg experiments/cfgs/faster_rcnn_end2end.yml --flip
```

### Getting Final Predictions

```
python get_final_predictions.py
```
