import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')


class ARIMAModel:
    def __init__(self):
        self.model = None
        self.order = (1, 1, 1)  # Basit model parametreleri
        self.predictions = None

    def fit_predict(self, train_data, test_data, prediction_length):
        try:
            # Geçmiş tahminler için boş liste
            all_predictions = []

            # Her 5 nokta için bir tahmin yap
            step_size = 5
            for i in range(0, len(test_data), step_size):
                # Eğitim verisini güncelle
                current_train = np.concatenate([train_data, test_data[:i]])

                # Model eğitimi
                model = ARIMA(current_train, order=self.order)
                model_fit = model.fit()

                # Sonraki 5 adım için tahmin
                pred = model_fit.forecast(steps=min(step_size, len(test_data) - i))
                all_predictions.extend(pred)

            # Eksik tahminleri tamamla
            if len(all_predictions) < len(test_data):
                all_predictions.extend([all_predictions[-1]] * (len(test_data) - len(all_predictions)))

            # Tahminleri numpy dizisine çevir
            self.predictions = np.array(all_predictions[:len(test_data)])

            # Tahminleri düzelt
            self.predictions = self.adjust_predictions(self.predictions, train_data[-1])

            # Son model eğitimi
            full_data = np.concatenate([train_data, test_data])
            self.model = ARIMA(full_data, order=self.order)
            self.model_fit = self.model.fit()

            return self.predictions

        except Exception as e:
            print(f"ARIMA fit_predict hatası: {e}")
            return np.zeros(len(test_data))

    def predict_future(self, data, steps):
        try:
            # Model eğitimi
            model = ARIMA(data, order=self.order)
            model_fit = model.fit()

            # Gelecek tahminler
            forecast = model_fit.forecast(steps=steps)

            # Tahminleri düzelt
            forecast = self.adjust_predictions(forecast, data[-1])

            # Tahminleri yumuşat
            forecast = self.smooth_predictions(forecast, data[-1])

            return forecast

        except Exception as e:
            print(f"ARIMA predict_future hatası: {e}")
            return np.zeros(steps)

    def adjust_predictions(self, predictions, last_value):
        """Tahminleri düzelt"""
        try:
            # Maksimum değişim
            max_change = 0.20  # %3 maksimum değişim

            # Alt ve üst sınırlar
            lower_bound = last_value * (1 - max_change)
            upper_bound = last_value * (1 + max_change)

            # Tahminleri sınırla
            predictions = np.clip(predictions, lower_bound, upper_bound)

            return predictions
        except Exception as e:
            print(f"Tahmin düzeltme hatası: {e}")
            return predictions

    def smooth_predictions(self, predictions, last_value, window=5):
        """Tahminleri yumuşat"""
        try:
            smoothed = predictions.copy()

            # İlk birkaç tahmin için yumuşak geçiş
            for i in range(min(window, len(predictions))):
                weight = (i + 1) / (window + 1)
                smoothed[i] = last_value * (1 - weight) + predictions[i] * weight

            return smoothed
        except Exception as e:
            print(f"Yumuşatma hatası: {e}")
            return predictions

    def calculate_r2_score(self, y_true, y_pred):
        try:
            return r2_score(y_true, y_pred) * 100
        except:
            return 0.0