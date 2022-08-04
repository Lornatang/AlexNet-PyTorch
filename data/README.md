# Usage

## Step1: Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver]()
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

## Step2: Prepare the dataset in the following format

```text
# Dataset struct
- MiniImageNet_1K
    - original
        - mini_imagenet
            - images
                - n0153282900000005.jpg
                - n0153282900000006.jpg
                - ...
        - train.csv
        - valid.csv
        - test.csv
```

## Step3: Preprocess the dataset

```bash
cd <AlexNet-PyTorch-main>/scripts
python3 preprocess_mini_imagenet.py
```

## Step4: Check that the final dataset directory schema is completely correct

```text
# Train dataset
- MiniImageNet_1K
    - train
       - n0153282900000005.jpg
       ...
    - valid
    - test
    - original
        - mini_imagenet
            - images
                - n0153282900000005.jpg
                - n0153282900000006.jpg
                - ...
        - train.csv
        - valid.csv
        - test.csv
```

