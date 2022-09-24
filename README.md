## ML-Decoder

Implementation for paper [ML-Decoder: Scalable and Versatile Classification Head](https://arxiv.org/abs/2111.12933).

The official pytorch [code](https://github.com/Alibaba-MIIL/ML_Decoder)

num group queries $K = \lceil \frac{num~classes}{group~factor} \rceil$

#### Sample Usage:

```
effnetv2 = EfficientNetV2B1(input_shape=[None, None, 3], include_top=False)
outputs = MLDecoder(num_classes=num_classes, d_model=128, dff=512, dropout_rate=0)(effnetv2.output)

model = Model(effnetv2.input, outputs)
```
Accuracy vs. Parameters and FLOPS for different classification heads on simple dataset.

<img src="https://github.com/bdghuy/ML-Decoder/blob/main/img_.PNG" width="404" height="212">
