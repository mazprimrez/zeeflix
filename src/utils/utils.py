import pandas as pd
import ast


def download_dataset():
    dataset = pd.read_csv('dataset/HR_result.csv')
    
    movies = pd.read_csv('dataset/simpler_movie_dataset.csv')
    movies = movies.drop_duplicates(subset='id').set_index('id')

    users = pd.read_csv('dataset/user_metadata.csv', index_col='userId')
    return dataset, movies, users

def preprocess_data(dataset):
    return ast.literal_eval(dataset)

def table_prep(dataset, movies):
    dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])

    wanted_keys = ast.literal_eval(dataset['movieId'])
    wanted_keys = [ele for ele in wanted_keys]
    mvs = dictfilt(movies, wanted_keys)
    meta = pd.DataFrame.from_dict(mvs, orient='index')
    df = pd.DataFrame({'movieId': [ele for ele in ast.literal_eval(dataset['movieId'])], 'rating': ast.literal_eval(dataset['rating'])}).set_index('movieId')
    df = df.merge(meta, left_index=True, right_index=True)
    df.sort_values('rating', ascending=False, inplace=True)
    return df

def get_reco_metadata(dataset, movies):
    dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])

    # get need movie id to be joined
    wanted_keys = ast.literal_eval(dataset['movies'])
    wanted_keys = [ele for ele in wanted_keys]
    mvs = dictfilt(movies, wanted_keys)

    meta = pd.DataFrame.from_dict(mvs, orient='index')
    df = pd.DataFrame({'movieId': [ele for ele in ast.literal_eval(dataset['movies'])], 'pred': ast.literal_eval(dataset['pred'])}).set_index('movieId')
    df = df.merge(meta, left_index=True, right_index=True)
    df.sort_values('pred', ascending=False, inplace=True)
    return df

def update_input_dropwdown(dataset):
    dataset = pd.DataFrame(dataset)
    return [{"label": title, "value": id} for title, id in zip(dataset['title'], dataset['movies'])]