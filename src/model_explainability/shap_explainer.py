# src/model_explainability/shap_explainer.py

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ModelExplainer:
    """
    SHAP değerleri kullanarak model tahminlerini açıklar.
    """
    def __init__(self, model, X_train):
        """
        Args:
            model: Eğitilmiş model
            X_train: Eğitim verisi
        """
        self.model = model
        self.X_train = X_train
        self.explainer = None

    def create_explainer(self):
        """SHAP explainer oluşturur."""
        try:
            self.explainer = shap.TreeExplainer(self.model)
            print("SHAP explainer başarıyla oluşturuldu.")
        except Exception as e:
            print(f"SHAP explainer oluşturulurken hata: {e}")

    def explain_prediction(self, X_test):
        """
        Belirli bir tahmin için SHAP değerlerini hesaplar.

        Args:
            X_test: Açıklanacak tahmin verisi

        Returns:
            shap_values: SHAP değerleri
        """
        if self.explainer is None:
            self.create_explainer()

        try:
            shap_values = self.explainer.shap_values(X_test)
            return shap_values
        except Exception as e:
            print(f"SHAP değerleri hesaplanırken hata: {e}")
            return None

    def plot_feature_importance(self, X_test):
        """
        Özellik önemlilik grafiğini çizer.

        Args:
            X_test: Test verisi
        """
        shap_values = self.explain_prediction(X_test)
        if shap_values is not None:
            plt.figure()
            shap.summary_plot(shap_values, X_test)
            plt.tight_layout()
            plt.show()

    def generate_explanation_report(self, X_test):
        """
        Tahmin açıklama raporu oluşturur.

        Args:
            X_test: Test verisi

        Returns:
            dict: Açıklama raporu
        """
        shap_values = self.explain_prediction(X_test)
        if shap_values is None:
            return None

        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(shap_values).mean(0)
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        return {
            'feature_importance': feature_importance.to_dict(),
            'shap_values': shap_values.tolist(),
            'mean_shap_value': np.mean(np.abs(shap_values))
        }