# Description: This is a simple dashboard example using Dash

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from flask import Flask

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("My Dashboard"),
                        html.P("This is my dashboard."),
                        dcc.Graph(
                            id='example-graph',
                            figure={
                                'data': [
                                    {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                                    {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
                                ],
                                'layout': {
                                    'title': 'Dash Data Visualization'
                                }
                            }
                        )
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Card Header"),
                                dbc.CardBody("This is some card text."),
                            ],
                        ),
                    ],
                    md=8,
                ),
            ]
        ),
    ],
    fluid=True,
)


if __name__ == '__main__':
    app.run_server(debug=True)
    