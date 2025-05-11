<div align="center">
<br>



## Get Started

### Training

```
./scripts/train.sh  --data_path [/path/to/train_data] --eval_data_path [/path/to/eval_data] --resnet_path [/path/to/pretrained_resnet_path] --convnext_path [/path/to/pretrained_convnext_path] --output_dir [/path/to/output_dir] [other args]
```

For example, training on ProGAN, run the following command:

```
./scripts/train.sh --data_path dataset/progan/train --eval_data_path dataset/progan/eval --resnet_path pretrained_ckpts/resnet50.pth --convnext_path pretrained_ckpts/open_clip_pytorch_model.bin --output_dir results/progan_train
```


