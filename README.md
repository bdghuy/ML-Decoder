## ML-Decoder

Implementation for paper [ML-Decoder: Scalable and Versatile Classification Head](https://arxiv.org/abs/2111.12933).

The official pytorch [code](https://github.com/Alibaba-MIIL/ML_Decoder)


$\text{num group queries } K = \lceil \frac{\text{num classes } C}{\text{group factor } g} \rceil$

#### Sample Usage:

```
effv2 = EfficientNetV2B0(include_top=False, 
                         input_shape=[None, None, 3])
logits = MLDecoder(num_classes=num_classes,
                   d_model=256,
                   dff=1024,
                   group_factor=10,
                   dropout_rate=0)(effv2.output)
outputs = tf.keras.layers.Softmax()(logits)

model = Model(effv2.input, outputs)
```
Accuracy vs. Parameters and FLOPS for different classification heads on simple dataset.

<img src="https://github.com/bdghuy/ML-Decoder/blob/main/img_.PNG" width="404" height="212">
