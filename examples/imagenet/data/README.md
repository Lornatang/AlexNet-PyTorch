### Imagenet

To run on Imagenet, place your `train` and `test` directories in `data`. 

Example commands: 
```bash
# Evaluate small AlexNet on CPU
python main.py data -e -a alexnet --pretrained 
```
```bash
# Evaluate medium AlexNet on GPU
python main.py data -e -a alexnet --pretrained --gpu 0 --batch-size 128
```
```bash
# Evaluate ResNet-50 for comparison
python main.py data -e -a resnet50 --pretrained --gpu 0
```
