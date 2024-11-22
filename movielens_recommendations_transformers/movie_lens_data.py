import json
import os
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd
import tensorflow as tf


class MovieLensData(object):
    _RAW_DATA = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    _DATA_FOLDER = "data"
    _LOCAL_TEST_DATA = f"{_DATA_FOLDER}/movielens.zip"
    _USERS_DATA = f"{_DATA_FOLDER}/ml-1m/users.dat"
    _RATINGS_DATA = f"{_DATA_FOLDER}/ml-1m/ratings.dat"
    _MOVIES_DATA = f"{_DATA_FOLDER}/ml-1m/movies.dat"
    _TRAIN_DATA = f"{_DATA_FOLDER}/train_data.csv"
    _TEST_DATA = f"{_DATA_FOLDER}/test_data.csv"
    _BATCH_SIZE = 256
    _SEQUENCE_LENGTH = 4
    _STEP_SIZE = 2

    USER_FEATURES = ["sex", "age_group", "occupation"]
    MOVIE_FEATURES = ["genres"]

    users = None
    ratings = None
    movies = None

    genres = [
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime" "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical" "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    def __init__(self):
        self._download_raw_data()

        if not os.path.exists(self._TRAIN_DATA) or not os.path.exists(self._TEST_DATA):
            self._preprocess()
            self._build_vocabularies()
            self._build_genre_vector()

    def _download_raw_data(self):
        if not os.path.exists(self._DATA_FOLDER):
            os.makedirs(self._DATA_FOLDER)

        if not os.path.exists(self._LOCAL_TEST_DATA):
            urlretrieve(self._RAW_DATA, self._LOCAL_TEST_DATA)
            ZipFile(self._LOCAL_TEST_DATA, "r").extractall(self._DATA_FOLDER)

    def _build_vocabularies(self):
        vocabularies = {
            "user_id": list(self.users.user_id.unique()),
            "movie_id": list(self.movies.movie_id.unique()),
            "sex": list(self.users.sex.unique()),
            "age_group": list(self.users.age_group.unique()),
            "occupation": list(self.users.occupation.unique()),
        }

        # write vocabularies to file in JSON format
        with open(f"{self._DATA_FOLDER}/vocabularies.json", "w") as f:
            json.dump(vocabularies, f)

    def _build_genre_vector(self):
        genre_vector = self.movies[self.genres].to_numpy().tolist()

        # write genre vector to file in JSON format
        with open(f"{self._DATA_FOLDER}/genre_vector.json", "w") as f:
            json.dump(genre_vector, f)

    def _preprocess(self):
        self.users = pd.read_csv(
            self._USERS_DATA,
            sep="::",
            names=["user_id", "sex", "age_group", "occupation", "zip_code"],
            encoding="ISO-8859-1",
            engine="python",
        )

        self.ratings = pd.read_csv(
            self._RATINGS_DATA,
            sep="::",
            names=["user_id", "movie_id", "rating", "unix_timestamp"],
            encoding="ISO-8859-1",
            engine="python",
        )

        self.movies = pd.read_csv(
            self._MOVIES_DATA,
            sep="::",
            names=["movie_id", "title", "genres"],
            encoding="ISO-8859-1",
            engine="python",
        )

        self.users["user_id"] = self.users["user_id"].apply(lambda x: f"user_{x}")
        self.users["age_group"] = self.users["age_group"].apply(lambda x: f"group_{x}")
        self.users["occupation"] = self.users["occupation"].apply(
            lambda x: f"occupation_{x}"
        )

        self.movies["movie_id"] = self.movies["movie_id"].apply(lambda x: f"movie_{x}")

        self.ratings["movie_id"] = self.ratings["movie_id"].apply(
            lambda x: f"movie_{x}"
        )
        self.ratings["user_id"] = self.ratings["user_id"].apply(lambda x: f"user_{x}")
        self.ratings["rating"] = self.ratings["rating"].apply(lambda x: float(x))

        for genre in self.genres:
            self.movies[genre] = self.movies["genres"].apply(
                lambda values: int(genre in values.split("|"))
            )

        ratings_group = self.ratings.sort_values(by=["unix_timestamp"]).groupby(
            "user_id"
        )

        ratings_data = pd.DataFrame(
            data={
                "user_id": list(ratings_group.groups.keys()),
                "movie_ids": list(ratings_group.movie_id.apply(list)),
                "ratings": list(ratings_group.rating.apply(list)),
                "timestamps": list(ratings_group.unix_timestamp.apply(list)),
            }
        )

        def create_sequences(values, window_size, step_size):
            sequences = []
            start_index = 0
            while True:
                end_index = start_index + window_size
                seq = values[start_index:end_index]
                if len(seq) < window_size:
                    seq = values[-window_size:]
                    if len(seq) == window_size:
                        sequences.append(seq)
                    break
                sequences.append(seq)
                start_index += step_size
            return sequences

        ratings_data.movie_ids = ratings_data.movie_ids.apply(
            lambda ids: create_sequences(ids, self._SEQUENCE_LENGTH, self._STEP_SIZE)
        )

        ratings_data.ratings = ratings_data.ratings.apply(
            lambda ids: create_sequences(ids, self._SEQUENCE_LENGTH, self._STEP_SIZE)
        )

        del ratings_data["timestamps"]

        ratings_data_movies = ratings_data[["user_id", "movie_ids"]].explode(
            "movie_ids", ignore_index=True
        )
        ratings_data_rating = ratings_data[["ratings"]].explode(
            "ratings", ignore_index=True
        )
        ratings_data_transformed = pd.concat(
            [ratings_data_movies, ratings_data_rating], axis=1
        )
        ratings_data_transformed = ratings_data_transformed.join(
            self.users.set_index("user_id"), on="user_id"
        )
        ratings_data_transformed.movie_ids = ratings_data_transformed.movie_ids.apply(
            lambda x: ",".join(x)
        )
        ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
            lambda x: ",".join([str(v) for v in x])
        )

        del ratings_data_transformed["zip_code"]

        ratings_data_transformed.rename(
            columns={"movie_ids": "sequence_movie_ids", "ratings": "sequence_ratings"},
            inplace=True,
        )

        random_selection = np.random.rand(len(ratings_data_transformed.index)) <= 0.85
        train_data = ratings_data_transformed[random_selection]
        test_data = ratings_data_transformed[~random_selection]

        train_data.to_csv(self._TRAIN_DATA, index=False, sep="|")
        test_data.to_csv(self._TEST_DATA, index=False, sep="|")

    def _get_dataset_from_csv(
        self, csv_file_path, shuffle=False, batch_size=_BATCH_SIZE
    ):
        def process(features):
            movie_ids_string = features["sequence_movie_ids"]
            sequence_movie_ids = tf.strings.split(movie_ids_string, ",").to_tensor()

            # The last movie id in the sequence is the target movie.
            features["sequence_movie_ids"] = sequence_movie_ids[:, :-1]

            ratings_string = features["sequence_ratings"]
            sequence_ratings = tf.strings.to_number(
                tf.strings.split(ratings_string, ","), tf.dtypes.float32
            ).to_tensor()

            # The last rating in the sequence is the target for the model to predict.
            target = sequence_ratings[:, -1]
            features["sequence_ratings"] = sequence_ratings[:, :-1]

            features["target_movie_id"] = sequence_movie_ids[:, -1]

            # !!! Daisuke Added following three lines !!!
            # for key, value in features.items():
            #     if len(value.shape) == 1 or (
            #         len(value.shape) == 2 and value.shape[1] == 1
            #     ):
            #         features[key] = tf.expand_dims(value, -1)

            return features, target

        dataset = tf.data.experimental.make_csv_dataset(
            csv_file_path,
            batch_size=batch_size,
            num_epochs=1,
            field_delim="|",
            shuffle=shuffle,
        ).map(process)

        return dataset

    def get_train_data(self):
        # tf.executing_eagerly()
        return self._get_dataset_from_csv(self._TRAIN_DATA)

    def get_test_data(self):
        return self._get_dataset_from_csv(self._TEST_DATA)

    def get_vocabularies(self):
        with open(f"{self._DATA_FOLDER}/vocabularies.json", "r") as f:
            return json.load(f)

    def get_genre_vectors(self):
        with open(f"{self._DATA_FOLDER}/genre_vector.json", "r") as f:
            return np.array(json.load(f))


if __name__ == "__main__":
    data = MovieLensData()
    data.get_train_data()
