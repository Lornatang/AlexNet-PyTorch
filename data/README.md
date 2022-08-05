# Usage

## Step1: Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver]()
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

## Step2: Prepare the dataset in the following format

```text
# Dataset struct
- ImageNet_1K
    - ILSVRC2012_img_train
        - ILSVRC2012_img_train.tar
    - ILSVRC2012_img_val
        - ILSVRC2012_img_val.tar
        - valprep.sh
```

## Step3: Preprocess the dataset

```bash
cd <AlexNet-PyTorch-main>/scripts
bash preprocess_imagenet.sh
```

## Step4: Check that the final dataset directory schema is completely correct

```text
# Train dataset
- ImageNet_1K
    - ILSVRC2012_img_train
        - n01440764
            - n01440764_18.JPEG
        - ...
    - ILSVRC2012_img_val
        - n01440764
            - ILSVRC2012_val_00000293.JPEG
        - ...
```

