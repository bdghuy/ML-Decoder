# ML-Decoder

[paper](https://arxiv.org/abs/2111.12933)

The official pytorch [code](https://github.com/Alibaba-MIIL/ML_Decoder)

### Sample Usage:

```
effnetv2 = EfficientNetV2B1(input_shape=[None, None, 3], include_top=False)
outputs = MLDecoder(num_classes=num_classes, d_model=128, dff=512, dropout_rate=0)(effnetv2.output)

model = Model(effnetv2.input, outputs)
```

<img src="https://github.com/bdghuy/ML-Decoder/blob/main/img.PNG" width="237" height="158">
