import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from models.base_model import BaseModel


class LSTMModel(BaseModel):
    def __init__(self):
        super().__init__()  # BaseModel'in __init__ metodunu çağır
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        self.sequence_length = 60
        self.r2_score = 0

    def prepare_data(self, data):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

    def fit_predict(self, train_data, test_data, prediction_length):
        try:
            # Veriyi hazırla
            X_train, y_train = self.prepare_data(train_data)

            # Modeli eğit
            self.model.fit(X_train, y_train)

            # Test verisi için tahmin
            full_data = np.concatenate((train_data, test_data))
            scaled_full = self.scaler.transform(full_data.reshape(-1, 1))

            X_test = []
            for i in range(len(train_data), len(full_data)):
                X_test.append(scaled_full[i - self.sequence_length:i, 0])

            X_test = np.array(X_test)
            predictions = self.model.predict(X_test)
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

            # R2 skorunu hesapla
            self.r2_score = r2_score(test_data, predictions) * 100

            return predictions

        except Exception as e:
            print(f"LSTM tahmin hatası: {e}")
            return np.zeros(len(test_data))

    def predict_future(self, data, steps):
        try:
            scaled_data = self.scaler.transform(data.reshape(-1, 1))
            last_sequence = scaled_data[-self.sequence_length:]

            predictions = []
            current_sequence = last_sequence.flatten()

            for _ in range(steps):
                next_pred = self.model.predict(current_sequence.reshape(1, -1))
                predictions.append(next_pred[0])
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = next_pred

            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions).flatten()

            return predictions

        except Exception as e:
            print(f"LSTM gelecek tahmin hatası: {e}")
            return np.zeros(steps)

    def calculate_r2_score(self, y_true, y_pred):
        try:
            return r2_score(y_true, y_pred) * 100
        except:
            return 0.0

    def save_state(self):
        """Model durumunu kaydet"""
        state = super().save_state()  # BaseModel'in save_state metodunu çağır
        model_state = {
            'model': self.model if hasattr(self, 'model') else None,
            'sequence_length': self.sequence_length if hasattr(self, 'sequence_length') else None
        }
        return {**state, **model_state}

    def load_state(self, state):
        """Model durumunu yükle"""
        super().load_state(state)  # BaseModel'in load_state metodunu çağır
        if state.get('model') is not None:
            self.model = state['model']
        if state.get('sequence_length') is not None:
            self.sequence_length = state['sequence_length']
        return self