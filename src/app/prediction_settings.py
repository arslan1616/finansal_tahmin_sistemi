# src/app/prediction_settings.py

import tkinter as tk
from tkinter import ttk
import json
import os

class PredictionSettings:
    """
    Tahmin ayarları ve uyarı sistemi için kullanıcı arayüzü.
    """
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Tahmin Ayarları")
        self.root.geometry("400x500")
        self.short_term_days = 30  # 1 aylık tahmin
        self.mid_term_days = 90  # 3 aylık tahmin
        self.long_term_days = 360  # 12 aylık tahmin

        # Ayarları yükle
        self.settings = self.load_settings()

        # Tahmin aralığı ayarları
        self.create_prediction_range_settings()

        # Fiyat uyarı ayarları
        self.create_price_alert_settings()

        # Kaydet butonu
        tk.Button(self.root, text="Ayarları Kaydet", command=self.save_settings).pack(pady=20)

    def get_prediction_range(self, range_type='short'):
        if range_type == 'short':
            return self.short_term_days
        elif range_type == 'mid':
            return self.mid_term_days
        else:
            return self.long_term_days

    def create_prediction_range_settings(self):
        """Tahmin aralığı ayarları için arayüz oluşturur."""
        frame = ttk.LabelFrame(self.root, text="Tahmin Aralığı Ayarları", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        # Gün sayısı seçimi
        tk.Label(frame, text="Tahmin günü sayısı:").pack()
        self.days_var = tk.StringVar(value=self.settings.get('prediction_days', '30'))
        days_entry = ttk.Entry(frame, textvariable=self.days_var)
        days_entry.pack()

        # Güven aralığı seçimi
        tk.Label(frame, text="Güven aralığı (%):").pack()
        self.confidence_var = tk.StringVar(value=self.settings.get('confidence_interval', '95'))
        confidence_entry = ttk.Entry(frame, textvariable=self.confidence_var)
        confidence_entry.pack()

    def create_price_alert_settings(self):
        """Fiyat uyarı ayarları için arayüz oluşturur."""
        frame = ttk.LabelFrame(self.root, text="Fiyat Uyarı Ayarları", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        # Üst limit uyarısı
        tk.Label(frame, text="Üst limit uyarısı (TL):").pack()
        self.upper_limit_var = tk.StringVar(value=self.settings.get('upper_limit', ''))
        upper_limit_entry = ttk.Entry(frame, textvariable=self.upper_limit_var)
        upper_limit_entry.pack()

        # Alt limit uyarısı
        tk.Label(frame, text="Alt limit uyarısı (TL):").pack()
        self.lower_limit_var = tk.StringVar(value=self.settings.get('lower_limit', ''))
        lower_limit_entry = ttk.Entry(frame, textvariable=self.lower_limit_var)
        lower_limit_entry.pack()

    def load_settings(self):
        """Kayıtlı ayarları yükler."""
        try:
            with open('settings.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_settings(self):
        """Ayarları kaydeder."""
        settings = {
            'prediction_days': self.days_var.get(),
            'confidence_interval': self.confidence_var.get(),
            'upper_limit': self.upper_limit_var.get(),
            'lower_limit': self.lower_limit_var.get()
        }

        with open('settings.json', 'w') as f:
            json.dump(settings, f)

    def run(self):
        """Uygulamayı çalıştırır."""
        self.root.mainloop()

if __name__ == "__main__":
    app = PredictionSettings()
    app.run()