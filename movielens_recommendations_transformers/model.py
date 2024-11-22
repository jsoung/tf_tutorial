import os
from collections import OrderedDict

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
from encoder import Encoder
from keras import layers
from movie_lens_data import MovieLensData


class Model(object):
    data = MovieLensData()
    sequence_length = data._SEQUENCE_LENGTH
    include_user_id = False
    include_user_features = False
    include_movie_features = False

    hidden_units = [256, 128]
    dropout_rate = 0.1
    num_heads = 3

    encoder = Encoder()

    def create_model_inputs(self):
        input_features = OrderedDict()

        input_features["user_id"] = keras.Input(
            name="user_id", shape=(1,), dtype="string"
        )
        input_features["sequence_movie_ids"] = keras.Input(
            name="sequence_movie_ids", shape=(self.sequence_length - 1,), dtype="string"
        )
        input_features["sequence_ratings"] = keras.Input(
            name="sequence_ratings", shape=(self.sequence_length - 1,), dtype=tf.float32
        )
        input_features["sex"] = keras.Input(name="sex", shape=(1,), dtype="string")
        input_features["age_group"] = keras.Input(
            name="age_group", shape=(1,), dtype="string"
        )
        input_features["occupation"] = keras.Input(
            name="occupation", shape=(1,), dtype="string"
        )
        input_features["target_movie_id"] = keras.Input(
            name="target_movie_id", shape=(1,), dtype="string"
        )

        return input_features

    def create_model(self):
        inputs = self.create_model_inputs()
        transformer_features, other_features = self.encoder.encode_input_features(
            inputs,
            self.include_user_id,
            self.include_user_features,
            self.include_movie_features,
        )

        # Create a multi-headed attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=transformer_features.shape[2],
            dropout=self.dropout_rate,
        )(transformer_features, transformer_features)

        # Transformer block.
        attention_output = layers.Dropout(self.dropout_rate)(attention_output)
        x1 = layers.Add()([transformer_features, attention_output])
        x1 = layers.LayerNormalization()(x1)
        x2 = layers.LeakyReLU()(x1)
        x2 = layers.Dense(units=x2.shape[-1])(x2)
        x2 = layers.Dropout(self.dropout_rate)(x2)
        transformer_features = layers.Add()([x1, x2])
        transformer_features = layers.LayerNormalization()(transformer_features)
        features = layers.Flatten()(transformer_features)

        # Included the other features.
        if other_features is not None:
            features = layers.concatenate(
                [features, layers.Reshape([other_features.shape[-1]])(other_features)]
            )

        # Fully-connected layers.
        for num_units in self.hidden_units:
            features = layers.Dense(num_units)(features)
            features = layers.BatchNormalization()(features)
            features = layers.LeakyReLU()(features)
            features = layers.Dropout(self.dropout_rate)(features)

        outputs = layers.Dense(units=1)(features)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model


if __name__ == "__main__":
    m = Model()
    model = m.create_model()

    model.compile(
        optimizer=keras.optimizers.Adagrad(learning_rate=0.01),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )

    train_dataset = m.data.get_train_data()
    model.fit(train_dataset, epochs=5)
    model.save_weights(f"{m.data._DATA_FOLDER}/model.weights.h5")

    test_dataset = m.data.get_test_data()
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test MAE: {round(rmse, 3)}")

    model.load_weights(f"{m.data._DATA_FOLDER}/model.weights.h5")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test MAE from saved model: {round(rmse, 3)}")
