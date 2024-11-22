import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from model import Model
from movie_lens_data import MovieLensData

data = MovieLensData()
m = Model()
model = m.create_model()

model.compile(
    optimizer=keras.optimizers.Adagrad(learning_rate=0.01),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()],
)

model.load_weights(f"{m.data._DATA_FOLDER}/model.weights.h5")

test_dataset = data.get_test_data()
_, rmse = model.evaluate(test_dataset, verbose=0)
print(f"Test MAE: {round(rmse, 3)}")
