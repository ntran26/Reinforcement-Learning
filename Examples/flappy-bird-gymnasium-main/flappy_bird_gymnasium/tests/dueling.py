import tensorflow as tf


class DuelingDQN(tf.keras.Model):
    def __init__(self, action_space):
        super(DuelingDQN, self).__init__()

        self.fc1 = tf.keras.layers.Dense(
            512,
            activation="elu",
            kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
        )
        self.fc2 = tf.keras.layers.Dense(
            256,
            activation="elu",
            kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
        )
        self.V = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
        )
        self.A = tf.keras.layers.Dense(
            action_space,
            activation=None,
            kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
        )

    def call(self, inputs, training=None):
        x = self.fc1(inputs, training=training)
        x = self.fc2(x, training=training)
        V = self.V(x, training=training)
        A = self.A(x, training=training)
        adv_mean = tf.reduce_mean(A, axis=-1, keepdims=True)
        return V + (A - adv_mean)

    def get_action(self, state):
        q_value = self(state)
        return tf.math.argmax(q_value, axis=-1)[0]
