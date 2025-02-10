import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import Orthogonal, TruncatedNormal
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Lambda,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)


class PositionalEmbedding(Layer):
    def __init__(self, units, dropout_rate=0.0, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)

        self.units = units
        self.projection = Dense(units, kernel_initializer=TruncatedNormal(stddev=0.02))
        self.dropout = Dropout(rate=dropout_rate)

    def build(self, input_shape):
        super(PositionalEmbedding, self).build(input_shape)

        self.position = self.add_weight(
            name="position",
            shape=(1, input_shape[1], self.units),
            initializer=TruncatedNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, inputs, training):
        x = self.projection(inputs)
        x = x + self.position

        return self.dropout(x, training=training)


class Encoder(Layer):
    def __init__(
        self,
        embed_dim,
        ff_mult,
        num_heads,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        **kwargs
    ):
        super(Encoder, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=attention_dropout_rate,
            kernel_initializer=TruncatedNormal(stddev=0.02),
        )

        self.dense_0 = Dense(
            units=embed_dim * ff_mult,
            activation="gelu",
            kernel_initializer=TruncatedNormal(stddev=0.02),
        )
        self.dense_1 = Dense(
            units=embed_dim, kernel_initializer=TruncatedNormal(stddev=0.02)
        )

        self.dropout_0 = Dropout(rate=dropout_rate)
        self.dropout_1 = Dropout(rate=dropout_rate)

        self.norm_0 = LayerNormalization(epsilon=1e-6)
        self.norm_1 = LayerNormalization(epsilon=1e-6)

        self.add_0 = Add()
        self.add_1 = Add()

    def call(self, inputs, training):
        # Attention block
        x = self.norm_0(inputs)
        x, attn_scores = self.mha(
            query=x,
            value=x,
            key=x,
            training=training,
            return_attention_scores=True,
        )
        x = self.dropout_0(x, training=training)
        x = self.add_0([x, inputs])

        # MLP block
        y = self.norm_1(x)
        y = self.dense_0(y)
        y = self.dense_1(y)
        y = self.dropout_1(y, training=training)

        return self.add_1([x, y]), attn_scores


class DuelingDQN(Model):
    def __init__(
        self, action_space, num_layers, embed_dim, ff_mult, num_heads, **kwargs
    ):
        super(DuelingDQN, self).__init__(**kwargs)

        # Input
        self.pos_embs = PositionalEmbedding(embed_dim)

        # Encoder
        self.e_layers = [
            Encoder(embed_dim, ff_mult, num_heads) for _ in range(num_layers)
        ]

        # Reduce
        # self.flatten = Lambda(lambda x: x[:, -1])
        # self.flatten = GlobalMaxPooling1D()
        self.flatten = GlobalAveragePooling1D()

        # Output
        self.V = Dense(
            1,
            activation=None,
            kernel_initializer=Orthogonal(0.01),
        )
        self.A = Dense(
            action_space,
            activation=None,
            kernel_initializer=Orthogonal(0.01),
        )

    def call(self, inputs, training=None):
        x = self.pos_embs(inputs, training=training)

        for layer in self.e_layers:
            x, attn_matrix = layer(x, training=training)

        # Reduce block
        x = self.flatten(x, training=training)
        # x = self.drop_out(x, training=training)

        # compute value & advantage
        V = self.V(x, training=training)
        A = self.A(x, training=training)

        # advantages have zero mean
        A -= tf.reduce_mean(A, axis=-1, keepdims=True)  # [B, A]

        return V + A, attn_matrix  # [B, A]

    def get_action(self, state):
        y, attn_matrix = self(state, training=False)
        return tf.math.argmax(y, axis=-1)[0], attn_matrix
