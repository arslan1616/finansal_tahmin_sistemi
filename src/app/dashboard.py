import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import sys
import os
import yaml
import warnings
from dash import dash_table

warnings.filterwarnings('ignore')

# Modelleri import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lstm_model import LSTMModel
from models.random_forest import RandomForestModel
from models.arima_model import ARIMAModel


class StockDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.config = self.load_config()
        self.stock_symbol = self.config['symbols'][0]
        self.available_symbols = self.config['symbols']
        self.prediction_days = 30
        self.models = {}
        self.predictions = {}
        self.future_predictions = {}
        self.model_cache = {}

        self.settings = {
            'prediction_ranges': {
                'short': 7,
                'medium': 15,
                'long': 30
            },
            'models': {
                'lstm': True,
                'rf': True,
                'arima': True
            },
            'data_paths': self.config['paths']
        }

        self.styles = {
            'container': {
                'margin': '0 auto',
                'padding': '20px',
                'max-width': '1200px'
            },
            'header': {
                'textAlign': 'center',
                'color': '#2c3e50',
                'marginBottom': '30px'
            },
            'dropdown': {
                'marginBottom': '20px'
            },
            'chart': {
                'marginBottom': '30px',
                'backgroundColor': 'white',
                'padding': '15px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            },
            'metrics': {
                'marginTop': '30px',
                'padding': '20px',
                'backgroundColor': 'white',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            }
        }

        # Gerekli klasörleri oluştur
        self.create_directories()

        # Layout ve callback'leri ayarla
        self.setup_layout()
        self.setup_callbacks()

    def load_config(self):
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config.yaml"
            )
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Config yükleme hatası: {e}")
            return {
                'symbols': ['GARAN.IS', 'AKBNK.IS', 'YKBNK.IS', 'THYAO.IS'],
                'paths': {
                    'raw_data': 'data/raw',
                    'processed_data': 'data/processed',
                    'models': 'saved_models'
                }
            }

    def load_data(self, symbol):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2 * 365)

            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)

            if len(df) < 100:
                raise ValueError(f"{symbol} için yeterli veri bulunamadı")

            self.stock_data = df[['Close']].copy()
            self.stock_data.fillna(method='ffill', inplace=True)

            return True
        except Exception as e:
            print(f"Veri yükleme hatası ({symbol}): {e}")
            return False

    def prepare_models(self, symbol, force_new=False):
        try:
            if not force_new and symbol in self.model_cache:
                self.models = self.model_cache[symbol]['models'].copy()
                self.predictions = self.model_cache[symbol]['predictions'].copy()
                return True

            self.models = {}
            self.predictions = {}

            if symbol not in self.model_cache:
                self.model_cache[symbol] = {'models': {}, 'predictions': {}}

            data = self.stock_data['Close'].values
            train_size = int(len(data) * 0.8)
            train_data = data[:train_size]
            test_data = data[train_size:]

            model_classes = {
                'lstm': LSTMModel,
                'rf': RandomForestModel,
                'arima': ARIMAModel
            }

            for model_name, model_class in model_classes.items():
                try:
                    model = model_class()
                    predictions = model.fit_predict(train_data, test_data, len(test_data))

                    self.models[model_name] = model
                    self.predictions[model_name] = predictions
                    self.model_cache[symbol]['models'][model_name] = model
                    self.model_cache[symbol]['predictions'][model_name] = predictions
                except Exception as e:
                    print(f"{model_name.upper()} model hatası: {e}")
                    self.models[model_name] = None
                    self.predictions[model_name] = np.zeros(len(test_data))
                    self.model_cache[symbol]['models'][model_name] = None
                    self.model_cache[symbol]['predictions'][model_name] = np.zeros(len(test_data))

            return True
        except Exception as e:
            print(f"Model hazırlama hatası: {e}")
            return False

    def create_directories(self):
        directories = [
            'data/raw',
            'data/processed',
            'saved_models'
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        for symbol in self.available_symbols:
            os.makedirs(f"saved_models/{symbol.replace('.', '_')}", exist_ok=True)

    def setup_layout(self):
        self.app.layout = html.Div([
            html.Div([
                html.H1(
                    id='stock-title',
                    children=f"{self.stock_symbol} Hisse Analizi",
                    style=self.styles['header']
                ),
                dcc.Dropdown(
                    id='stock-selector',
                    options=[{'label': s, 'value': s} for s in self.available_symbols],
                    value=self.stock_symbol,
                    style={'width': '300px'}
                ),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),

            html.Div([
                html.Label("Tahmin Aralığı:"),
                dcc.Dropdown(
                    id='prediction-range',
                    options=[
                        {'label': '7 Gün', 'value': 7},
                        {'label': '15 Gün', 'value': 15},
                        {'label': '30 Gün', 'value': 30}
                    ],
                    value=30,
                    style=self.styles['dropdown']
                ),
            ], style={'marginBottom': '20px'}),

            html.Div([
                html.Label("Tahmin Modeli:"),
                dcc.Dropdown(
                    id='model-selector',
                    options=[
                        {'label': 'LSTM', 'value': 'lstm'},
                        {'label': 'Random Forest', 'value': 'rf'},
                        {'label': 'ARIMA', 'value': 'arima'},
                        {'label': 'Tüm Modeller', 'value': 'all'}
                    ],
                    value='all',
                    style=self.styles['dropdown']
                )
            ]),

            html.Div(id='error-message', style={'display': 'none'}),

            dcc.Tabs([
                dcc.Tab(label='Tahmin Grafikleri', children=[
                    dcc.Graph(
                        id='prediction-chart',
                        style=self.styles['chart']
                    ),
                    html.Div([
                        html.H3("Model Başarı Oranları", style={'textAlign': 'center'}),
                        html.Div(id='success-rates')
                    ], style=self.styles['metrics'])
                ]),

                dcc.Tab(label='Model Performans Analizi', children=[
                    html.Div([
                        html.H3("Model Performans Analizi"),
                        html.Div([
                            html.Label("Model Seçimi:"),
                            dcc.Dropdown(
                                id='performance-model-selector',
                                options=[
                                    {'label': 'LSTM', 'value': 'lstm'},
                                    {'label': 'Random Forest', 'value': 'rf'},
                                    {'label': 'ARIMA', 'value': 'arima'},
                                    {'label': 'Tüm Modeller', 'value': 'all'}
                                ],
                                value='all',
                                style={'width': '200px', 'margin': '10px 0'}
                            )
                        ]),

                        html.Div([
                            html.Label("Minimum Model Başarı Oranı (%)"),
                            dcc.Slider(
                                id='model-success-rate-slider',
                                min=0,
                                max=100,
                                step=5,
                                value=60,
                                marks={i: f'{i}%' for i in range(0, 101, 10)}
                            )
                        ], style={'margin': '20px 0'}),

                        html.Button('Analizi Güncelle', id='refresh-performance-analysis',
                                    style={'margin': '10px 0'}),

                        dash_table.DataTable(
                            id='model-performance-table',
                            columns=[
                                {'name': 'Hisse', 'id': 'Hisse'},
                                {'name': 'Model', 'id': 'Model'},
                                {'name': 'Başarı Oranı (%)', 'id': 'Success_Rate'},
                                {'name': 'Son Tahmin (TL)', 'id': 'Last_Prediction'},
                                {'name': 'Potansiyel Getiri (%)', 'id': 'Potential_Return'}
                            ],
                            style_table={'height': '400px', 'overflowY': 'auto'},
                            style_cell={'textAlign': 'center'},
                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold'
                            },
                            style_data_conditional=[
                                {
                                    'if': {
                                        'filter_query': '{Success_Rate} >= 80'
                                    },
                                    'backgroundColor': 'rgba(0, 255, 0, 0.1)'
                                }
                            ],
                            sort_action='native',
                            filter_action='native'
                        )
                    ], style=self.styles['metrics'])
                ])
            ]),
        ], style=self.styles['container'])

    def setup_callbacks(self):
        @self.app.callback(
            [Output('prediction-chart', 'figure'),
             Output('success-rates', 'children'),
             Output('stock-title', 'children'),
             Output('error-message', 'children')],
            [Input('stock-selector', 'value'),
             Input('model-selector', 'value'),
             Input('prediction-range', 'value')]
        )
        def update_charts(selected_stock, selected_model, prediction_days):
            try:
                if selected_stock != self.stock_symbol:
                    self.stock_symbol = selected_stock
                    if not self.load_data(selected_stock):
                        raise ValueError(f"{selected_stock} için veri yüklenemedi")
                    self.prepare_models(selected_stock)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=self.stock_data.index,
                    y=self.stock_data['Close'],
                    name='Gerçek Değer',
                    line=dict(color='#000000', width=2)
                ))

                colors = {
                    'lstm': '#FF4B4B',
                    'rf': '#2E93fA',
                    'arima': '#66DA26'
                }

                models_to_show = ['lstm', 'rf', 'arima'] if selected_model == 'all' else [selected_model]
                success_rates = []

                for model_name in models_to_show:
                    if self.models.get(model_name):
                        if len(self.predictions[model_name]) > 0:
                            fig.add_trace(go.Scatter(
                                x=self.stock_data.index[-len(self.predictions[model_name]):],
                                y=self.predictions[model_name],
                                name=f'{model_name.upper()} Tahmin',
                                line=dict(color=colors[model_name], width=1.5, dash='dot')
                            ))

                        try:
                            future_dates = pd.date_range(
                                start=self.stock_data.index[-1] + pd.Timedelta(days=1),
                                periods=prediction_days,
                                freq='D'
                            )

                            future_pred = self.models[model_name].predict_future(
                                self.stock_data['Close'].values,
                                prediction_days
                            )

                            last_value = self.stock_data['Close'].values[-1]
                            future_pred = self.smooth_predictions(last_value, future_pred, prediction_days)

                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=future_pred,
                                name=f'{model_name.upper()} Gelecek Tahmin',
                                line=dict(color=colors[model_name], width=2)
                            ))

                            r2_score = self.models[model_name].calculate_r2_score(
                                self.stock_data['Close'].values[-len(self.predictions[model_name]):],
                                self.predictions[model_name]
                            )
                        except Exception as e:
                            print(f"Tahmin hatası ({model_name}): {e}")
                            r2_score = 0

                        success_rates.append(self.create_metric_box(
                            model_name, r2_score, colors[model_name]
                        ))

                fig.update_layout(
                    height=600,
                    title=dict(
                        text=f'{selected_stock} Hisse Senedi Fiyat Tahminleri',
                        x=0.5,
                        xanchor='center'
                    ),
                    xaxis_title='Tarih',
                    yaxis_title='Fiyat (TL)',
                    hovermode='x unified',
                    template='plotly_white',
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(255, 255, 255, 0.8)'
                    )
                )

                success_rates_container = html.Div(
                    success_rates,
                    style={
                        'display': 'flex',
                        'justifyContent': 'center',
                        'flexWrap': 'wrap'
                    }
                )

                return fig, success_rates_container, f"{selected_stock} Hisse Analizi", ""

            except Exception as e:
                print(f"Genel hata: {e}")
                return {}, [], f"{selected_stock} Hisse Analizi", str(e)

        @self.app.callback(
            Output('model-performance-table', 'data'),
            [Input('refresh-performance-analysis', 'n_clicks'),
             Input('performance-model-selector', 'value'),
             Input('model-success-rate-slider', 'value')],
            prevent_initial_call=True
        )
        def update_performance_table(n_clicks, selected_model, min_success_rate):
            if n_clicks is None:
                return dash.no_update

            # State'i kaydet
            original_state = self.save_current_state()
            performance_data = []

            try:
                for symbol in self.available_symbols:
                    if not self.load_data(symbol):
                        continue

                    if symbol not in self.model_cache:
                        if not self.prepare_models(symbol, force_new=True):
                            continue

                    current_price = float(self.stock_data['Close'].iloc[-1])
                    models_to_check = ['lstm', 'rf', 'arima'] if selected_model == 'all' else [selected_model]

                    for model_name in models_to_check:
                        if model_name in self.model_cache[symbol]['models'] and self.model_cache[symbol]['models'][
                            model_name]:
                            try:
                                model = self.model_cache[symbol]['models'][model_name]
                                predictions = self.model_cache[symbol]['predictions'][model_name]

                                r2_score = model.calculate_r2_score(
                                    self.stock_data['Close'].values[-len(predictions):],
                                    predictions
                                )

                                if r2_score >= min_success_rate:
                                    future_pred = model.predict_future(
                                        self.stock_data['Close'].values,
                                        30
                                    )
                                    future_pred = self.smooth_predictions(
                                        self.stock_data['Close'].values[-1],
                                        future_pred
                                    )
                                    last_prediction = float(future_pred[-1])
                                    potential_return = ((last_prediction - current_price) / current_price) * 100

                                    performance_data.append({
                                        'Hisse': symbol,
                                        'Model': model_name.upper(),
                                        'Success_Rate': f"{r2_score:.2f}",
                                        'Last_Prediction': f"{last_prediction:.2f}",
                                        'Potential_Return': f"{potential_return:.2f}"
                                    })

                            except Exception as e:
                                print(f"{symbol} {model_name} performans hesaplama hatası: {e}")
                                continue

            finally:
                # State'i geri yükle
                self.restore_state(original_state)

            performance_data.sort(key=lambda x: float(x['Success_Rate']), reverse=True)
            return performance_data

    def save_current_state(self):
        return {
            'symbol': self.stock_symbol,
            'stock_data': self.stock_data.copy() if hasattr(self, 'stock_data') else None,
            'models': {k: v.save_state() if v and hasattr(v, 'save_state') else v
                       for k, v in self.models.items()} if hasattr(self, 'models') else {},
            'predictions': {k: v.copy() for k, v in self.predictions.items()} if hasattr(self, 'predictions') else {},
            'cache': {k: {'models': {m: v.save_state() if v and hasattr(v, 'save_state') else v
                                     for m, v in c['models'].items()},
                          'predictions': c['predictions'].copy() if 'predictions' in c else {}}
                      for k, c in self.model_cache.items()}
        }

    def restore_state(self, state):
        self.stock_symbol = state['symbol']
        if state['stock_data'] is not None:
            self.stock_data = state['stock_data'].copy()

        self.models = {k: v.load_state(state['models'][k]) if v and hasattr(v, 'load_state') else v
                       for k, v in self.models.items()}
        self.predictions = {k: v.copy() for k, v in state['predictions'].items()}

        self.model_cache = {k: {'models': {m: v.load_state(state['cache'][k]['models'][m])
        if v and hasattr(v, 'load_state') else v
                                           for m, v in c['models'].items()},
                                'predictions': state['cache'][k]['predictions'].copy()
                                if 'predictions' in state['cache'][k] else {}}
                            for k, c in state['cache'].items()}

    def smooth_predictions(self, last_actual, predictions, window=5):
        smooth_pred = predictions.copy()
        for i in range(min(window, len(predictions))):
            weight = (i + 1) / (window + 1)
            smooth_pred[i] = last_actual * (1 - weight) + predictions[i] * weight
        return smooth_pred

    def create_metric_box(self, model_name, score, color):
        return html.Div([
            html.H4(f"{model_name.upper()}"),
            html.H2(f"%{score:.1f}"),
            html.P("Başarı Oranı", style={'fontSize': '12px'})
        ], style={
            'textAlign': 'center',
            'margin': '10px',
            'padding': '15px',
            'backgroundColor': color,
            'color': 'white',
            'borderRadius': '8px',
            'width': '150px'
        })

    def run(self):
        print("\nMevcut Hisse Listesi:")
        print("-" * 30)
        for symbol in self.available_symbols:
            print(f"- {symbol}")
        print("\nDashboard başlatılıyor... http://localhost:8050")

        if self.load_data(self.stock_symbol):
            self.prepare_models(self.stock_symbol)

        self.app.run_server(debug=False, port=8050)


if __name__ == "__main__":
    dashboard = StockDashboard()
    dashboard.run()