import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from MPICentroidKMeans import MPICentroidKMeans
from SerialKMeans import SerialKMeans

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')


def prepare_data():
    df = pd.read_json('./dataset.json', lines=True)
    df = df.drop(['link', 'headline', 'authors', 'date'], axis=1)
    enc = sbert_model.encode(df['short_description'])
    return enc


if __name__ == '__main__':
    encodings = prepare_data()
