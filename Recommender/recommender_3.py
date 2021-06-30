import requests
import io
import numpy as np
import scipy.sparse as sp
import pandas as pd

from lightfm.datasets import _common
from lightfm import LightFM

def _read_data(path):
    url = path
    s = requests.get(url).content
    data = pd.read_csv(io.StringIO(s.decode('utf-8')), sep='\t', header=None)
    return preprocess(data)

def _read_local_data(path):
    data = pd.read_csv(path, sep='\t', header=None)
    return preprocess(data)

def preprocess(data):
    #열 이름 바꾸기
    data.columns = ['uid', 'iid', 'rating', 'timeStamp']
    #timestamp 버리기
    data.drop(columns='timeStamp', inplace=True)
    return data

def _get_dimensions(train_data, test_data):
    uids = set(train_data['uid'])|set(test_data['uid'])
    iids = set(train_data['iid'])|set(test_data['iid'])

    rows = len(uids)
    cols = len(iids)

    return rows, cols

def _build_interaction_matrix(rows, cols, data, min_rating):

    mat = sp.lil_matrix((rows, cols), dtype=np.int32)
    val = data.values
    for uid, iid, rating in val:
        if rating >= min_rating:
            mat[uid-1, iid-1] = rating #id값에서 1씩 빼줌

    return mat.tocoo()

def _parse_item_metadata(num_items, item_metadata_raw, genres_raw):

    genres = []

    for line in genres_raw:
        if line:
            genre, gid = line.split("|")
            genres.append("genre:{}".format(genre))

    id_feature_labels = np.empty(num_items, dtype=np.object)
    genre_feature_labels = np.array(genres)

    id_features = sp.identity(num_items, format="csr", dtype=np.float32)
    genre_features = sp.lil_matrix((num_items, len(genres)), dtype=np.float32)

    for line in item_metadata_raw:

        if not line:
            continue

        splt = line.split("|")

        # Zero-based indexing
        iid = int(splt[0]) - 1
        title = splt[1]

        id_feature_labels[iid] = title

        item_genres = [idx for idx, val in enumerate(splt[5:]) if int(val) > 0]

        for gid in item_genres:
            genre_features[iid, gid] = 1.0

    return (
        id_features,
        id_feature_labels,
        genre_features.tocsr(),
        genre_feature_labels,
    )

def fetch_amazonbooks(
    indicator_features = True,
    genre_features = False,
    min_rating = 0.0,
    download_if_missing = True,
):
    if not (indicator_features or genre_features):
        raise ValueError(
            "At least one of item_indicator_features " "or genre_features must be True"
        )

    #Load raw data
    #train_raw = _read_data('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Books.csv')
    #test_raw = _read_data('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Books.csv')

    train_raw = _read_local_data('u1.base')
    test_raw = _read_local_data('u1.test')

    # Figure out the dimensions
    num_users, num_items = _get_dimensions(train_raw, test_raw)

    # Load train interactions
    train = _build_interaction_matrix(
        num_users, num_items, train_raw, min_rating
    )
    # Load test interactions
    test = _build_interaction_matrix(num_users, num_items, test_raw, min_rating)

    assert train.shape == test.shape

    #여기부터!!!!!
    item_metadata_raw = None
    genres_raw = None
    # Load metadata features
    (
        id_features,
        id_feature_labels,
        genre_features_matrix,
        genre_feature_labels,
    ) = _parse_item_metadata(num_items, item_metadata_raw, genres_raw)

    assert id_features.shape == (num_items, len(id_feature_labels))
    assert genre_features_matrix.shape == (num_items, len(genre_feature_labels))

    if indicator_features and not genre_features:
        features = id_features
        feature_labels = id_feature_labels
    elif genre_features and not indicator_features:
        features = genre_features_matrix
        feature_labels = genre_feature_labels
    else:
        features = sp.hstack([id_features, genre_features_matrix]).tocsr()
        feature_labels = np.concatenate((id_feature_labels, genre_feature_labels))

    data = {
        "train": train,
        "test": test,
        "item_features": features,
        "item_feature_labels": feature_labels,
        "item_labels": id_feature_labels,
    }

    return data

#fetch data and format it
data = fetch_amazonbooks(min_rating=4.0) #우리는 4 이상의 data만 받는다

#create model
model = LightFM(loss='warp')#Weighted Approximate-Rank Pairwise

#train model
model.fit(data['train'], epochs=30, num_threads=2)

def recommendation(model, data):
    #number of users and movies in training data
    n_users, n_items = data['train'].shape

    user_ids = list(range(0,n_users))
    item_ids = list(range(0,len(data['item_labels'])))

    #generate recommendations for each user we input
    top_items = dict()
    for user_id in user_ids:
        #movies they already like
        known_positives = np.array(item_ids)[data['train'].tocsr()[user_id].indices]

        #movies our model predics they will like
        scores = model.predict(user_id, np.arange(n_items))
        top_items[user_id] = np.array(item_ids)[np.argsort(-scores)]
        top_items[user_id] = np.array([x for x in top_items[user_id] if x not in known_positives])

    return top_items
            
recommendation(model, data)