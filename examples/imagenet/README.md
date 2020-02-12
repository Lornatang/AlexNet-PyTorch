### Imagenet

To run on Imagenet, place your `train` and `val` directories in `data`. 

Example commands: 
```bash
# Evaluate AlexNet on CPU
python main.py data -e -a alexnet --pretrained 
```
```bash
# Evaluate AlexNet on GPU
python main.py data -e -a alexnet --pretrained --gpu 0 --batch-size 128
```
```bash
# Evaluate ResNet-50 for comparison
python main.py data -e -a resnet50 --pretrained --gpu 0
```
