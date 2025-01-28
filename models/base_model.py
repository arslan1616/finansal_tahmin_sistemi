# base_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class BaseModel:
    def calculate_r2_score(self, y_true, y_pred):
        """R2 skoru hesaplama"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = max(0, 1 - (ss_res / (ss_tot + 1e-10)))
        return r2 * 100

    def add_technical_features(self, data):
        """Gelişmiş teknik indikatörler"""
        # Veriyi düzleştir ve Series'e çevir
        if isinstance(data, np.ndarray):
            if len(data.shape) > 1:
                data = data.flatten()
        df = pd.DataFrame(data, columns=['close'])

        # Temel teknik indikatörler
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()

        # Momentum indikatörleri
        df['momentum'] = df['close'].pct_change(periods=5)
        df['roc'] = df['close'].pct_change(periods=12) * 100

        # Volatilite indikatörleri
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['SMA_20'] + (df['std_20'] * 2)
        df['lower_band'] = df['SMA_20'] - (df['std_20'] * 2)

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # NaN değerleri doldur
        df = df.fillna(method='bfill').fillna(method='ffill')

        return df.values

    def smooth_predictions(self, predictions, window=5):
        """Tahminleri yumuşatma"""
        return pd.Series(predictions).rolling(window=window, min_periods=1, center=True).mean().values