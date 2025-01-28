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

        # Klasörleri oluştur
        self.create_directories()

        # Model cache'i ekle
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

        self.setup_layout()
        self.setup_callbacks()

    def load_config(self):
        try:
            # Proje kök dizininden config.yaml'ı yükle
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
        """Veri yükleme fonksiyonu güncellendi"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2 * 365)

            # Yahoo Finance'dan veriyi çek
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)

            if len(df) < 100:
                raise ValueError(f"{symbol} için yeterli veri bulunamadı")

            # Sadece gerekli sütunları al ve NaN değerleri temizle
            self.stock_data = df[['Close']].copy()
            self.stock_data.fillna(method='ffill', inplace=True)

            print(f"Veri yüklendi: {symbol}, Satır sayısı: {len(self.stock_data)}")
            return True

        except Exception as e:
            print(f"Veri yükleme hatası ({symbol}): {e}")
            return False

    def prepare_models(self, symbol):
        """Model hazırlama fonksiyonu"""
        try:
            # Cache kontrolü - hem modelleri hem tahminleri kontrol et
            if symbol in self.model_cache and 'models' in self.model_cache[symbol] and 'predictions' in \
                    self.model_cache[symbol]:
                print(f"{symbol} için cache'den modeller ve tahminler yükleniyor...")
                self.models = self.model_cache[symbol]['models'].copy()
                self.predictions = self.model_cache[symbol]['predictions'].copy()
                return True

            print(f"{symbol} için yeni modeller hazırlanıyor...")
            self.models = {}
            self.predictions = {}

            # Cache için yeni yapı oluştur
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
                    print(f"{model_name.upper()} modeli hazırlanıyor...")
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

    def calculate_potential_returns(self):
        """Tüm hisseler için potansiyel getirileri hesapla"""
        potential_returns = []

        # Mevcut durumu sakla
        current_state = {
            'symbol': self.stock_symbol,
            'models': self.models.copy() if hasattr(self, 'models') else {},
            'predictions': self.predictions.copy() if hasattr(self, 'predictions') else {},
            'stock_data': self.stock_data.copy() if hasattr(self, 'stock_data') else None
        }

        try:
            for symbol in self.available_symbols:
                try:
                    # Veriyi yükle
                    if not self.load_data(symbol):
                        continue

                    # Modelleri hazırla veya cache'den al
                    if symbol not in self.model_cache:
                        if not self.prepare_models(symbol):
                            continue
                    else:
                        self.models = self.model_cache[symbol]['models'].copy()
                        self.predictions = self.model_cache[symbol]['predictions'].copy()

                    # Son kapanış fiyatı
                    current_price = self.stock_data['Close'].iloc[-1]

                    # Her model için tahminleri al
                    predictions = {}
                    for model_name in ['lstm', 'rf', 'arima']:
                        if model_name in self.models and self.models[model_name]:
                            try:
                                future_pred = self.models[model_name].predict_future(
                                    self.stock_data['Close'].values,
                                    30
                                )
                                predictions[model_name] = future_pred[-1]
                            except Exception as e:
                                print(f"{symbol} {model_name} tahmin hatası: {e}")
                                continue

                    # Ortalama tahmin
                    if predictions:
                        avg_prediction = sum(predictions.values()) / len(predictions)
                        change_percent = ((avg_prediction - current_price) / current_price) * 100

                        potential_returns.append({
                            'Hisse': symbol,
                            'Mevcut Fiyat': f"{current_price:.2f}",
                            'Tahmin Edilen Fiyat': f"{avg_prediction:.2f}",
                            'Potansiyel Getiri (%)': f"{change_percent:.2f}",
                            'Raw_Return': change_percent
                        })

                except Exception as e:
                    print(f"{symbol} getiri hesaplama hatası: {e}")
                    continue

        finally:
            # Önceki durumu tam olarak geri yükle
            self.stock_symbol = current_state['symbol']
            self.models = current_state['models']
            self.predictions = current_state['predictions']
            if current_state['stock_data'] is not None:
                self.stock_data = current_state['stock_data']

            # Aktif hisse için modelleri ve tahminleri yeniden yükle
            if self.stock_symbol in self.model_cache:
                self.models = self.model_cache[self.stock_symbol]['models'].copy()
                self.predictions = self.model_cache[self.stock_symbol]['predictions'].copy()

        return pd.DataFrame(potential_returns)

    def setup_layout(self):
        """Layout güncellendi"""
        self.app.layout = html.Div([
            # Üst kısım - Başlık ve Hisse Seçici
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

            # Tahmin Aralığı Seçici
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

            # Model Seçici
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

            # Hata mesajları için gizli div
            html.Div(id='error-message', style={'display': 'none'}),

            # Tab'lar
            dcc.Tabs([
                # Tahmin Grafikleri Tab'ı
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

                # Potansiyel Getiri Analizi Tab'ı
                dcc.Tab(label='Potansiyel Getiri Analizi', children=[
                    html.Div([
                        html.H3("Yükseliş Potansiyeline Göre Hisseler"),

                        # Filtreleme kontrolü
                        html.Div([
                            html.Label("Minimum Getiri Potansiyeli (%)"),
                            dcc.Slider(
                                id='return-filter-slider',
                                min=0,
                                max=100,
                                step=5,
                                value=10,
                                marks={i: f'{i}%' for i in range(0, 101, 10)}
                            )
                        ], style={'margin': '20px 0'}),

                        # Yenileme butonu
                        html.Button('Analizi Güncelle', id='refresh-analysis',
                                    style={'margin': '10px 0'}),

                        # Tablo
                        dash_table.DataTable(
                            id='potential-returns-table',
                            columns=[
                                {'name': 'Hisse', 'id': 'Hisse'},
                                {'name': 'Mevcut Fiyat (TL)', 'id': 'Mevcut Fiyat'},
                                {'name': 'Tahmin Edilen Fiyat (TL)', 'id': 'Tahmin Edilen Fiyat'},
                                {'name': 'Potansiyel Getiri (%)', 'id': 'Potansiyel Getiri (%)'}
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
                                        'filter_query': '{Raw_Return} >= 20'
                                    },
                                    'backgroundColor': 'rgba(0, 255, 0, 0.1)'
                                },
                                {
                                    'if': {
                                        'filter_query': '{Raw_Return} <= -20'
                                    },
                                    'backgroundColor': 'rgba(255, 0, 0, 0.1)'
                                }
                            ],
                            sort_action='native',
                            filter_action='native'
                        )
                    ], style=self.styles['metrics'])
                ])
            ])
        ], style=self.styles['container'])

    def create_directories(self):
        """Gerekli klasörleri oluştur"""
        directories = [
            'data/raw',
            'data/processed',
            'saved_models'
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        # Her hisse için model klasörü
        for symbol in self.available_symbols:
            os.makedirs(f"saved_models/{symbol.replace('.', '_')}", exist_ok=True)

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
                # Hisse değiştiyse veriyi yeniden yükle
                if selected_stock != self.stock_symbol:
                    self.stock_symbol = selected_stock
                    if not self.load_data(selected_stock):
                        raise ValueError(f"{selected_stock} için veri yüklenemedi")
                    self.prepare_models(selected_stock)

                fig = go.Figure()

                # Gerçek değerler
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
                        # Geçmiş tahminler
                        if len(self.predictions[model_name]) > 0:
                            fig.add_trace(go.Scatter(
                                x=self.stock_data.index[-len(self.predictions[model_name]):],
                                y=self.predictions[model_name],
                                name=f'{model_name.upper()} Tahmin',
                                line=dict(color=colors[model_name], width=1.5, dash='dot')
                            ))

                        # Gelecek tahminler
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

                            # Tahminleri yumuşat
                            last_value = self.stock_data['Close'].values[-1]
                            if prediction_days > 15:
                                future_pred = self.smooth_long_term_predictions(
                                    last_value,
                                    future_pred,
                                    prediction_days
                                )
                            else:
                                future_pred = self.smooth_transition(
                                    last_value,
                                    future_pred
                                )

                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=future_pred,
                                name=f'{model_name.upper()} Gelecek Tahmin',
                                line=dict(color=colors[model_name], width=2)
                            ))

                            # Başarı oranını hesapla
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

                # Grafik düzeni
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
            Output('potential-returns-table', 'data'),
            [Input('refresh-analysis', 'n_clicks'),
             Input('return-filter-slider', 'value')],
            prevent_initial_call=True
        )
        def update_returns_table(n_clicks, min_return):
            # Mevcut durumu sakla
            current_stock = self.stock_symbol
            current_models = self.models.copy() if hasattr(self, 'models') else {}
            current_predictions = self.predictions.copy() if hasattr(self, 'predictions') else {}

            try:
                # Potansiyel getirileri hesapla
                df = self.calculate_potential_returns()

                # Minimum getiri filtresini uygula
                filtered_df = df[df['Raw_Return'] >= min_return]

                # Sonuçları hazırla
                results = filtered_df.to_dict('records')

            except Exception as e:
                print(f"Getiri tablosu güncelleme hatası: {e}")
                results = []

            finally:
                # Önceki durumu geri yükle
                self.stock_symbol = current_stock
                if current_stock in self.model_cache:
                    self.models = self.model_cache[current_stock]
                else:
                    self.models = current_models
                self.predictions = current_predictions

                # Aktif hisse için modelleri yeniden yükle
                if self.stock_symbol in self.model_cache:
                    self.prepare_models(self.stock_symbol)

            return results

    def smooth_transition(self, last_actual, predictions, window=5):
        """Tahminler için yumuşak geçiş"""
        smooth_predictions = predictions.copy()
        for i in range(min(window, len(predictions))):
            weight = (i + 1) / (window + 1)
            smooth_predictions[i] = last_actual * (1 - weight) + predictions[i] * weight
        return smooth_predictions

    def smooth_long_term_predictions(self, last_actual, predictions, days):
        """Uzun vadeli tahminler için yumuşatma"""
        smooth_predictions = predictions.copy()
        last_30_days = self.stock_data['Close'].values[-30:]
        trend = np.mean(np.diff(last_30_days)) / np.mean(last_30_days)

        for i in range(len(predictions)):
            weight = min(i / 15, 1.0)  # İlk 15 gün için kademeli geçiş
            trend_prediction = last_actual * (1 + trend * (i + 1))
            smooth_predictions[i] = predictions[i] * weight + trend_prediction * (1 - weight)

        return smooth_predictions

    def create_metric_box(self, model_name, score, color):
        """Başarı oranı kutusu oluştur"""
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
        """Dashboard'ı başlat"""
        print("\nMevcut Hisse Listesi:")
        print("-" * 30)
        for symbol in self.available_symbols:
            print(f"- {symbol}")
        print("\nDashboard başlatılıyor... http://localhost:8050")

        # İlk hisse için verileri yükle
        if self.load_data(self.stock_symbol):
            self.prepare_models(self.stock_symbol)

        self.app.run_server(debug=False, port=8050)


if __name__ == "__main__":
    dashboard = StockDashboard()
    dashboard.run()