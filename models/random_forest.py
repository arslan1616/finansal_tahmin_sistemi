import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


from models.base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__()  # BaseModel'in __init__ metodunu çağır
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.sequence_length = 60
        self.r2_score = 0

    def prepare_data(self, data):
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def fit_predict(self, train_data, test_data, prediction_length):
        try:
            X_train, y_train = self.prepare_data(train_data)
            self.model.fit(X_train, y_train)

            full_data = np.concatenate((train_data, test_data))
            X_test = []
            for i in range(len(train_data), len(full_data)):
                X_test.append(full_data[i - self.sequence_length:i])

            predictions = self.model.predict(np.array(X_test))
            self.r2_score = r2_score(test_data, predictions) * 100

            return predictions

        except Exception as e:
            print(f"RandomForest tahmin hatası: {e}")
            return np.zeros(len(test_data))

    def predict_future(self, data, steps):
        try:
            predictions = []
            current_sequence = data[-self.sequence_length:]

            for _ in range(steps):
                next_pred = self.model.predict(current_sequence.reshape(1, -1))
                predictions.append(next_pred[0])
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = next_pred

            return np.array(predictions)

        except Exception as e:
            print(f"RandomForest gelecek tahmin hatası: {e}")
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