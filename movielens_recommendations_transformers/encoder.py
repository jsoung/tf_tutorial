import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import math

import keras
import tensorflow as tf
from keras import layers
from movie_lens_data import MovieLensData

class Encoder(object):
    data = MovieLensData()
    sequense_length = data._SEQUENCE_LENGTH

    def encode_input_features(
        self,
        inputs,
        include_user_id=True,
        include_user_features=True,
        include_movie_features=True,
    ):
        encoded_transformer_features = []
        encoded_other_features = []

        other_feature_names = []
        if include_user_id:
            other_feature_names.append("user_id")
        if include_user_features:
            other_feature_names.extend(self.data.USER_FEATURES)

        ## Encode user features
        vocabularies = self.data.get_vocabularies()
        for feature_name in other_feature_names:
            # Convert the string input values into integer indices.
            vocabulary = vocabularies[feature_name]
            idx = layers.StringLookup(
                vocabulary=vocabulary, mask_token=None, num_oov_indices=0
            )(inputs[feature_name])
            # Compute embedding dimensions
            embedding_dims = int(math.sqrt(len(vocabulary)))
            # Create an embedding layer with the specified dimensions.
            embedding_encoder = layers.Embedding(
                input_dim=len(vocabulary),
                output_dim=embedding_dims,
                name=f"{feature_name}_embedding",
            )
            # Convert the index values to embedding representations.
            encoded_other_features.append(embedding_encoder(idx))

        ## Create a single embedding vector for the user features
        if len(encoded_other_features) > 1:
            encoded_other_features = layers.concatenate(encoded_other_features)
        elif len(encoded_other_features) == 1:
            encoded_other_features = encoded_other_features[0]
        else:
            encoded_other_features = None

        ## Create a movie embedding encoder
        movie_vocabulary = vocabularies["movie_id"]
        movie_embedding_dims = int(math.sqrt(len(movie_vocabulary)))
        # Create a lookup to convert string values to integer indices.
        movie_index_lookup = layers.StringLookup(
            vocabulary=movie_vocabulary,
            mask_token=None,
            num_oov_indices=0,
            name="movie_index_lookup",
        )
        # Create an embedding layer with the specified dimensions.
        movie_embedding_encoder = layers.Embedding(
            input_dim=len(movie_vocabulary),
            output_dim=movie_embedding_dims,
            name=f"movie_embedding",
        )
        # Create a vector lookup for movie genres.
        genre_vectors = self.data.get_genre_vectors()
        movie_genres_lookup = layers.Embedding(
            input_dim=genre_vectors.shape[0],
            output_dim=genre_vectors.shape[1],
            embeddings_initializer=keras.initializers.Constant(genre_vectors),
            trainable=False,
            name="genres_vector",
        )
        # Create a processing layer for genres.
        movie_embedding_processor = layers.Dense(
            units=movie_embedding_dims,
            activation="relu",
            name="process_movie_embedding_with_genres",
        )

        ## Define a function to encode a given movie id.
        def encode_movie(movie_id):
            # Convert the string input values into integer indices.
            movie_idx = movie_index_lookup(movie_id)
            movie_embedding = movie_embedding_encoder(movie_idx)
            encoded_movie = movie_embedding
            if include_movie_features:
                movie_genres_vector = movie_genres_lookup(movie_idx)
                encoded_movie = movie_embedding_processor(
                    layers.concatenate([movie_embedding, movie_genres_vector])
                )

            return encoded_movie

        ## Encoding target_movie_id
        target_movie_id = inputs["target_movie_id"]
        encoded_target_movie = encode_movie(target_movie_id)

        ## Encoding sequence movie_ids.
        sequence_movies_ids = inputs["sequence_movie_ids"]
        encoded_sequence_movies = encode_movie(sequence_movies_ids)
        # Create positional embedding.
        position_embedding_encoder = layers.Embedding(
            input_dim=self.sequense_length,
            output_dim=movie_embedding_dims,
            name="position_embedding",
        )
        positions = tf.range(start=0, limit=self.sequense_length - 1, delta=1)
        encodded_positions = position_embedding_encoder(positions)
        # Retrieve sequence ratings to incorporate them into the encoding of the movie.
        sequence_ratings = inputs["sequence_ratings"]
        sequence_ratings = keras.ops.expand_dims(sequence_ratings, -1)

        # Add the positional encoding to the movie encodings and multiply them by rating.
        encoded_sequence_movies_with_poistion_and_rating = layers.Multiply()(
            [(encoded_sequence_movies + encodded_positions), sequence_ratings]
        )

        # Construct the transformer inputs.
        for i in range(self.sequense_length - 1):
            feature = encoded_sequence_movies_with_poistion_and_rating[:, i, ...]
            feature = keras.ops.expand_dims(feature, 1)
            encoded_transformer_features.append(feature)

        encoded_transformer_features.append(encoded_target_movie)

        encoded_transformer_features = layers.concatenate(
            encoded_transformer_features, axis=1
        )

        return encoded_transformer_features, encoded_other_features
