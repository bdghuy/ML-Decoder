import tensorflow as tf 

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
               dropout_rate=0.1
               ):
    super(DecoderLayer, self).__init__()

    # Multi-head cross-attention.
    self.mha_cross = tf.keras.layers.MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=d_model, # Size of each attention head for query Q and key K.
        dropout=dropout_rate
    )

    # Point-wise feed-forward network.
    self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'),
                                    tf.keras.layers.Dense(d_model)])
    # Layer normalization.
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    # Dropout for the point-wise feed-forward network.
    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
    self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, enc_output, training):
    # The encoder output shape is `(batch_size, input_seq_len, d_model)`.

    out1 = self.dropout1(x, training=training)
    out1 = self.layernorm1(out1 + x)

    # Cross-attention.
    attn_cross = self.mha_cross(
        query=out1,
        value=enc_output,
        key=enc_output,
        training=training  
    )
    out2 = self.layernorm2(attn_cross + out1)

    # Point-wise feed-forward network output.
    ffn_output = self.ffn(out2)  
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)

    return out3

class MLDecoder(tf.keras.layers.Layer):
  def __init__(self,
               *,
               num_classes,
               d_model,
               dff,
               num_attention_heads=8,
               emb_seed=0,
               dropout_rate=0.1
               ):
    super(MLDecoder, self).__init__()

    emb_init = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=emb_seed)
    w_init = tf.keras.initializers.GlorotNormal()

    # projection layer
    self.project = tf.keras.Sequential([tf.keras.layers.Dense(d_model, activation='relu')])
    # decoder layer
    self.query_emb = emb_init(shape=(1, num_classes, d_model))
    self.decoder = DecoderLayer(d_model=d_model, num_attention_heads=num_attention_heads, dff=dff, dropout_rate=dropout_rate)
    # cls layer
    self.w = tf.Variable(w_init(shape=(1, d_model, 1)), trainable=True)
    self.softmax = tf.keras.layers.Softmax()

  def call(self, x, training):
    x = tf.reshape(x, [-1, tf.shape(x)[1]*tf.shape(x)[2], tf.shape(x)[3]])
    x = self.project(x)

    batch_queries = tf.tile(self.query_emb, multiples=[tf.shape(x)[0], 1, 1])
    x = self.decoder(batch_queries, x, training)

    x = tf.matmul(x, self.w)
    x = tf.squeeze(x, [-1])
    x = self.softmax(x)
    return x