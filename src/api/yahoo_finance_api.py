import yfinance as yf
import pandas as pd
import os
import yaml
from typing import List

# Ana proje dizinini belirle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_config() -> dict:
    """Ana proje dizinindeki config.yaml dosyasını yükler."""
    config_path = os.path.join(BASE_DIR, "config.yaml")  # config.yaml ana dizinde

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config dosyası bulunamadı: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_historical_data(ticker: str, start_date: str, end_date: str, save_dir: str) -> None:
    """Yahoo Finance API kullanarak hisse senedi verilerini çeker ve belirtilen dizine kaydeder."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            print(f"{ticker} için veri bulunamadı.")
            return

        # Klasörün tam yolunu BASE_DIR kullanarak oluştur
        full_save_dir = os.path.join(BASE_DIR, save_dir)
        os.makedirs(full_save_dir, exist_ok=True)

        # Dosya kaydedilecek tam yolu belirle
        filename = f"{ticker}_historical_data.csv"
        save_path = os.path.join(full_save_dir, filename)

        data.to_csv(save_path, index=True)
        print(f"{ticker} verisi başarıyla kaydedildi: {save_path}")

    except Exception as e:
        print(f"{ticker} işlenirken hata oluştu: {str(e)}")


def fetch_multiple_stocks() -> None:
    """Config dosyasındaki tüm hisseleri çeker ve verileri belirtilen dizine kaydeder."""
    config = load_config()

    fetch_multiple_stocks_logic(
        tickers=config["symbols"],
        start_date=config["date_range"]["start"],
        end_date=config["date_range"]["end"],
        save_dir=config["paths"]["raw_data"]
    )


def fetch_multiple_stocks_logic(tickers: List[str], start_date: str, end_date: str, save_dir: str) -> None:
    """Çekirdek mantık (test edilebilir versiyon)"""
    for ticker in tickers:
        fetch_historical_data(ticker, start_date, end_date, save_dir)


if __name__ == "__main__":
    fetch_multiple_stocks()
