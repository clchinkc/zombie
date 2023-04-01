


import dash_bootstrap_components as dbc
from dash import Dash, dcc, html
from flask import Flask
from views import init_dash_app, main_blueprint

app = Flask(__name__)
app.register_blueprint(main_blueprint)
dash_app = Dash(__name__, server=app, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
dash_app.layout = html.Div(id='empty-layout')
init_dash_app(dash_app)

if __name__ == "__main__":
    app.run(debug=True)
