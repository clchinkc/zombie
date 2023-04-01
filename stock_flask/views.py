


from candlestick import create_candlestick_figure, create_dash_layout
from flask import Blueprint, render_template, render_template_string, request
from prediction import predict_stock_price

main_blueprint = Blueprint('main', __name__)

def init_dash_app(dash_app):
    global _dash_app
    _dash_app = dash_app

@main_blueprint.route('/')
def home():
    return render_template('home.html')

@main_blueprint.route('/predict', methods=['POST'])
def predict():
    stock = request.form['stock']
    period = request.form['period']
    algorithm = request.form['algorithm']
    stock, future_y, plot_data = predict_stock_price(stock, period, algorithm)
    return render_template('prediction.html', stock=stock, future_price=future_y[-1], plot_data=plot_data)

@main_blueprint.route('/candlestick', methods=['POST'])
def candlestick():
    stock = request.form['stock']
    period = request.form['period']
    fig = create_candlestick_figure(stock, period)
    _dash_app.layout = create_dash_layout(fig)
    return render_template_string('''
    {{ app_entry|safe }}
    <footer></footer>
    {{ scripts|safe }}
    {{ renderer|safe }}
    ''', app_entry=_dash_app.index(), scripts=_dash_app._generate_scripts_html(),
    renderer=_dash_app._generate_renderer())


