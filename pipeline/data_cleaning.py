import pandas as pd
import numpy as np

df = pd.read_csv(
    "../data/ratings.csv",
    dtype={
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32",
    },
    usecols=["userId", "movieId", "rating"]
)

print("Cleaning data...")

'''
CREATE A CONFIDENCE COLUMN

    In this case confidence will be equal to rating, this is subject to change
'''

df['confidence'] = df['rating'].astype(np.float32)

'''
FILTER SPARSE USERS/MOVIES

    users with 1 rating -> meaningless embeddings
    movies with 1 rating -> random noise
    model overfits noise
    memory wasted on useless rows/cols
'''

min_user_interactions = 20
min_movie_interactions = 20

while True:
    before = len(df)
    user_counts = df["userId"].value_counts()
    movie_counts = df["movieId"].value_counts()
    df = df[df["userId"].map(user_counts) >= min_user_interactions]
    df = df[df["movieId"].map(movie_counts) >= min_movie_interactions]
    if len(df) == before:
        break

'''
ENCODE IDs INTO CONGIGUOUS INTEGERS
'''

df = df.reset_index(drop=True)

df["userId_internal"], user_index = pd.factorize(df["userId"])
df["movieId_internal"], movie_index = pd.factorize(df["movieId"])

df['userId_internal'] = df['userId_internal'].astype(np.int32)
df['movieId_internal'] = df['movieId_internal'].astype(np.int32)

'''
SAVE DATA INTO PARQUET FILES
'''

# full dataframe
df.to_parquet("../data/full_data.parquet", engine="pyarrow")

# ml dataframe
df[['userId_internal', 'movieId_internal', 'confidence']].to_parquet("../data/train_data.parquet", engine="pyarrow")

# mapping the old ids to new ids for future search
users = df[['userId', 'userId_internal']].drop_duplicates()
movies = df[['movieId', 'movieId_internal']].drop_duplicates()

users.to_parquet("../data/users.parquet", index=False)
movies.to_parquet("../data/movies.parquet", index=False)
