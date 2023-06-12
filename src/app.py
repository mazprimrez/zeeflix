import dash
from dash import Dash, dcc, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import numpy as np
import plotly.graph_objects as go
import json
from urllib.request import urlopen
from catboost import CatBoostClassifier, Pool
import shap

from utils.utils import download_dataset, table_prep, get_reco_metadata, update_input_dropwdown
from utils.graph import distribution, distribution_mrr, distribution_recall, preprocessing_shap, waterfall_plot, top_movies_table

model =CatBoostClassifier()
model.load_model('dataset/Catboost',           
           format="cbm")
explainer = shap.TreeExplainer(model)    


col_list = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count', 'movie_age', 'production_companies', 'production_countries', 'spoken_languages', 'genres', 'total_movies', 'userAvgRating', 'userAvgBudget', 'userAvgPopularity', 'userAvgMovieAge', 'userAvgRuntime', 'userTopPH', 'userTopGenres']
categorical_col = ['production_companies', 'production_countries', 'spoken_languages', 'genres', 'userTopPH', 'userTopGenres']


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, "assets/style.css"])
server = app.server

app.layout = html.Div([

    dbc.Button(id='refresh-data', n_clicks=0, children='⏳️', className= "d-grid float-box" ,color='primer', outline=True),

    dbc.Row([
        dbc.Col([
            html.H1("Movies Recommendation"),
            dbc.Button(id='randomize-user', n_clicks=0, children='Get a Random User', className= "d-grid gap-1 mx-auto m-3",color='success', outline=True)
            ])], justify='center'),

# ----- first page ------ #

    dbc.Row([
        dbc.Col(children=[
            html.H3('Get to Know the User!'),
            html.Div(id="text", className='p-2'),
            dcc.Graph(id='table-users-review')
        ], width=2, className='p-3 mx-auto page-container', style={'overflow-y': 'hidden'}),

        dbc.Col(id='reco-poster', className='page-container')
    ], className='mx-auto'),

    html.Hr(style={'border-top': '8px solid #bbb', 'border-radius': '5px'}, className='m-5'),

# ----- end of first page ------ #

# ----- second page ------ #

    dbc.Row([
        dbc.Row([
            html.H3('Model Explanation')
        ], className='p-3 mx-auto', justify='center'),
            dbc.Col([
                dbc.Row([
                    html.P("""
                    This multi-stage Recommendation System consists of two models. First is Candidate Generators utilising Tensorflow Recommenders. Second, from tensorflow recommenders, I ranked the model based on what users like the most using Catboost Classifier. I trained the model using users who reviewed 20-500 movies, there are about 3Mio rows dataset. I predicted 388 from 207K unique users and the model returned 1,090 unique movies (45K total movies). From both model, I got 11.3% MRR score for heavy ranker and 13.5% Recall@K score for Candidate Generator. Which is a quite good score for a recommendation system. Here's the distribution of the MRR and Recall@K score. We also know that Recall@K is the more the users rating history the lower recall score, yet there is no correlation between MRR score and users watch history.
                    """),
                    dbc.Row([
                        dbc.Col(dbc.Row(dcc.Graph(id='accuracy-distribution', style={'width':'90vw'})), width=4),
                        dbc.Col(dbc.Row(dcc.Graph(id='recall-distribution', style={'width':'90vw'})), width=4),
                        dbc.Col(dbc.Row(dcc.Graph(id='mrr-distribution', style={'width':'90vw'})), width=4)
                    ])
                    ])
                ], width=9),
            dbc.Col([
                html.H4('Some movies appear more than others'),
                dcc.Graph(figure=top_movies_table())
                ], width=3),
            html.Hr(style={'border-top': '8px solid #bbb', 'border-radius': '5px'}, className='m-5'),
            dbc.Row([
                dbc.Col([
                    html.H3('Global Explanation'),
                    dbc.Row(html.Img(src='assets/shap_summary.png')),

                ], ),
                dbc.Col([
                    html.H3('Local Explanation'),
                    dcc.Dropdown(id='update-dropdown',multi=False, clearable=False,disabled=False,),
                    html.Div(id = 'waterfall-chart')
                ])
            ]),
    ], className='page-container'),

    # ---- THIRD PAGE ---- #
    dbc.Row([
        dbc.Col([
        html.H5('''Things to be improved in the future.'''),
        html.P('Candidate Retrieval models could be optimized | SHAP shows feature dependencies for both numerical and categorical features | Model workflow | more descriptive explanation on movies and users | more intuitive features.'),
        html.Br(),
        html.H5('Meet the Author'),
        html.A(html.P('LinkedIn'), href='https://www.linkedin.com/in/maziprimareza/', target='_blank')
        ], style={'margin':'auto'}, className='mx-auto')
        ], className='page-container', justify='center', style={'text-align':'center'}),

    # ----- END OF THIRD PAGE -----_#

# ----- end of second page ------ #
    
    # Store dataset
    dcc.Store(id='user-info', storage_type='memory'),
    dcc.Store(id='users-meta', storage_type='memory'),
    dcc.Store(id='movies-meta', storage_type='memory'),
    dcc.Store(id='reco-info', storage_type='memory'),
    dcc.Store(id='shap-preprocess', storage_type='memory')
])

# ----- second page ------ #

@app.callback(
    Output('accuracy-distribution', 'figure'),
    Output('recall-distribution', 'figure'),
    Output('mrr-distribution', 'figure'),
    Input('reco-info', 'data')
)
def accuracy_distribution(dataset):
    return distribution(dataset=dataset), distribution_recall(dataset=dataset), distribution_mrr(dataset=dataset)

@app.callback(
    Output('shap-preprocess', 'data'),
    Input('user-info', 'data'),
    Input('movies-meta', 'data'),
    Input('users-meta', 'data')
)
def shap_processing(whole_data, movies, users):
    dataset = preprocessing_shap(whole_data, users, movies, col_list, categorical_col).to_dict('records')
    return dataset

@app.callback(
    Output('update-dropdown', 'options'),
    Input('shap-preprocess', 'data'),
)
def update_input_dropdown(data):
    return update_input_dropwdown(data)

@app.callback(
    Output('waterfall-chart', 'children'),
    Input('shap-preprocess', 'data'),
    Input('update-dropdown', 'value')
)
def waterfall(dataset, mvsId):
    try:
        return waterfall_plot(mvsId, dataset, explainer, col_list)
    except:
        return None


# ----- first page ------ #

#store dataset
@app.callback(
    Output('reco-info', 'data'),
    Output('movies-meta', 'data'),
    Output('users-meta', 'data'),
    Input('refresh-data', 'n_clicks')
)
def movies_metadata(value):
    reco, mvs, users = download_dataset()
    reco = reco.to_dict('records')
    mvs = mvs.to_dict('index')
    users = users.to_dict('index')
    return reco, mvs, users


@app.callback(
            Output('user-info', 'data'),
            Input('randomize-user', 'n_clicks'),
            Input('reco-info', 'data'))
def randomized_users(n_clicks, dataset):
    userInfo = np.random.choice(dataset)
    return userInfo


@app.callback(
    Output('text', 'children'),
    Input('user-info', 'data'),
    Input('users-meta', 'data')
)
def text(dataset, users_metadata):
    userId = str(dataset['userId'])
    return f"""The model could predict the user {userId}'s taste {round(dataset['recall@5000'],4)*100}% and ranked it with {round(dataset['MRR'],4)*100}% accuracy based on MRR calculation!
    The user usually give {users_metadata[userId]['userAvgRating']} stars rating and has reviewed {users_metadata[userId]['total_movies']} movies in total.
    Their favorite genre is {users_metadata[userId]['userTopGenres']} and mostly watch movies from {users_metadata[userId]['userTopPH']}.
    """


@app.callback(
    Output('table-users-review', 'figure'),
    Input('user-info', 'data'),
    Input('movies-meta', 'data')
)
def table_user_reviews(dataset, movies):
    df = table_prep(dataset, movies)
    fig = go.Figure(data=go.Table(
        header=dict(
            values=['<b>Movie Title<b>', '<b>⭐️Rating</b>'],
            font=dict(size=14, color='#683B2B'),
            align="center",
            fill_color='#DeD1BD'
        ),
        cells=dict(
            values=[df['title'], df['rating']],
            align = "left",
            fill_color='white',
            font=dict(size=12),
            line_color='#683B2B')))
    fig.update_layout(
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        margin = dict(l=0,r=0,b=1,t=5,pad=0),
        )
    return fig


@app.callback(
    Output('reco-poster', 'children'),
    Input('user-info', 'data'),
    Input('movies-meta', 'data')
)
def thumbnail_poster(dataset, movies):

    dataset = get_reco_metadata(dataset, movies)

    main_path = 'https://image.tmdb.org/t/p/w500'
    imdb_path = 'https://www.imdb.com/title/'

    api_key = "a28aff9f112861ea28a73f73b0220165"

    try:
        return dbc.Row([
            dbc.Col([
                dbc.Row(html.A(html.Img(src=main_path + str(json.loads(urlopen(f"https://api.themoviedb.org/3/movie/{imdb_id}?api_key={api_key}").read())['poster_path']), style={'width':'100%'}), href=imdb_path+imdb_id, target='_blank')),
                dbc.Row(f'{round(pred, 4)*100}%', className='pred-box mx-auto'),
                ], width=2, style={'display':'inline-block'}, className='p-0 m-0') for imdb_id, pred in zip(dataset['imdb_id'], dataset['pred'])
            ])
    except:
        dbc.Row([html.H2("Oops! Some errors just occured. Please try again!")])


# ----- end of first page ------ #


if __name__ == "__main__":
    app.run_server(debug=True)