import plotly.graph_objects as go
import shap
import pandas as pd
import ast
from io import BytesIO
import base64
from dash import html
import matplotlib.pyplot as plt

def top_movies_table():
        fig = go.Figure(data=go.Table(
        header=dict(
            values=['title', 'counts', 'accuracy', 'counts pct(%)', 'accuracy pct(%)'],
            font=dict(size=14, color='#683B2B'),
            align="center",
            fill_color='#DeD1BD',
            height=40
        ),
        columnwidth = [200,80,80,80,80],
        cells=dict(
            values=[['The Sixth Sense', 'Ride Lonesome', 'The Run of the Country', 'Minions', 'Vali', 'Lovelines', 
            'Apache Country', 'Home Made Home', 'The Gun That Won the West', 'Cousin, Cousine', 'Virtue', 
            'The Holy Modal Rounders: Bound to Lose', 'Stalker', 'The Haunted House', 'Duel of Hearts', '10 Items or Less', 
            'The Walking Stick', 'Furious 7', 'Oldboy', 'Scarface'], 
            [136, 132, 130, 87, 82, 75, 73, 62, 62, 60, 60, 59, 59, 58, 53, 53, 52, 50, 49, 49], 
            [21, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1, 0, 0, 0, 1, 0, 0, 0, 7], 
            [35.05, 34.02, 33.51, 22.42, 21.13, 19.33, 18.81, 15.98, 15.98, 15.46, 15.46, 
            15.21, 15.21, 14.95, 13.66, 13.66, 13.4, 12.89, 12.63, 12.63], 
            [5.41, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.03, 0.0, 0.26, 0.0, 0.0, 0.0, 0.26, 0.0, 0.0, 0.0, 1.8]],
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

def distribution(dataset):
    data = {'Heavy Ranker (MRR)': [x['MRR'] for x in dataset], 'Candidate Generator (Recall@K)': [x['recall@5000'] for x in dataset]}
    ops = [1, 0.5]
    colors = ['#3D5361', '#4C3949']
    fig = go.Figure()
    for data_line, color, op in zip(data.keys(), colors, ops):
        fig.add_trace(go.Violin(x=data[data_line], line_color=color, meanline_visible=True, name=data_line, opacity=op))

    fig.update_traces(orientation='h', side='positive', width=9, points=False)
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False, 
            plot_bgcolor = 'rgba(0, 0, 0, 0)',
            paper_bgcolor = 'rgba(0, 0, 0, 0)',
            margin = dict(l=0,r=0,b=1,t=50,pad=0),title_text="<b>Distribution of MRR and Recall@K Score<b>",
            showlegend=True,
            legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.6)
            )
    fig.update_yaxes(visible=False)
    return fig


def distribution_recall(dataset):
        data = {'len': [x['len'] for x in dataset], 'Recall': [x['recall@5000'] for x in dataset]}

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data['len'], y=data['Recall'], mode='markers',  marker_color='#3D5361', 
                                opacity=0.6,
                                name='Recall'))

        fig.update_traces(mode='markers', marker_line_width=0, marker_size=8)

        fig.update_layout(
                plot_bgcolor = 'rgba(0, 0, 0, 0)',
                paper_bgcolor = 'rgba(0, 0, 0, 0)',
                margin = dict(l=5,r=5,b=5,t=30,pad=0),
                xaxis_showgrid=False, yaxis_showgrid=False,
                xaxis_title="Number of User Watch Histories",
                yaxis_title="Recall@K",
                legend=dict(
                        yanchor="top",
                        y=0.6,
                        xanchor="left",
                        x=0.8),
                title=dict(
                        text="Correlation of Recall and Number of<br>Users Watch History"
                )
                )

        return fig

def distribution_mrr(dataset):
        data = {'len': [x['len'] for x in dataset], 'MRR': [x['MRR'] for x in dataset]}

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data['len'], y=data['MRR'], mode='markers', marker_color='#4C3949', name='MRR'))

        fig.update_traces(mode='markers', marker_line_width=0, marker_size=8)

        fig.update_layout(
                plot_bgcolor = 'rgba(0, 0, 0, 0)',
                paper_bgcolor = 'rgba(0, 0, 0, 0)',
                margin = dict(l=5,r=5,b=5,t=30,pad=0),
                xaxis_showgrid=False, yaxis_showgrid=False,
                xaxis_title="Number of User Watch Histories",
                yaxis_title="MRR",
                legend=dict(
                        yanchor="top",
                        y=0.6,
                        xanchor="left",
                        x=0.8),
                title=dict(
                        text="Correlation of MRR and Number of<br>Users Watch History"
                )
                )

        return fig

def preprocessing_shap(dataset, users, movies, col_list, categorical_col):
        user_dataset = pd.DataFrame(dataset, index=[0])
        user_dataset['movies'] = user_dataset['movies'].apply(ast.literal_eval)
        user_dataset['userId'] = user_dataset['userId'].astype('int')
        user_dataset = user_dataset.explode('movies')[['userId', 'movies']]
        user_dataset['movies'] = user_dataset['movies'].astype('int')
        user_info = user_dataset['userId'].apply(lambda x: users[str(x)]).apply(pd.Series)
        mvs_info = user_dataset['movies'].apply(lambda x: movies[str(x)]).apply(pd.Series)
        dataset = pd.concat([user_dataset, user_info, mvs_info], axis=1).reset_index()
        dataset[categorical_col] = dataset[categorical_col].fillna('other')
        dataset[col_list] = dataset[col_list].fillna(-1)
        return dataset

def waterfall_plot(mvsId, dataset, explainer, col_list):
        dataset = pd.DataFrame(dataset)
        idx = dataset[dataset['movies']==mvsId].index[0]
        shap_values = explainer(dataset[col_list])
        shap_values.display_data = dataset[col_list].values
        plt.figure()
        waterfall = shap.plots.waterfall(shap_values[idx], show=False)
        waterfall.tight_layout()

        buffer = BytesIO()
        waterfall.savefig(buffer, format='png')
        buffer.seek(0)

        image_string = base64.b64encode(buffer.getvalue()).decode()

        return html.Div([html.Img(src='data:image/png;base64,' + image_string)])
        