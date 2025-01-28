import pandas as pd
import numpy as np
import os
import yaml


def load_config():
    """Config.yaml dosyasını yükler."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def detect_outliers(symbol: str, processed_data_dir: str, reports_dir: str) -> str:
    """
    Hisse bazlı aykırı değer analizi yapar

    Parameters:
    - symbol (str): Hisse sembolü
    - processed_data_dir (str): İşlenmiş veri klasörü
    - reports_dir (str): Rapor klasörü

    Returns:
    str: Rapor dosya yolu
    """
    try:
        # Dosya yollarını oluştur
        input_path = os.path.join(processed_data_dir, f"{symbol}_processed.csv")
        output_path = os.path.join(reports_dir, f"{symbol}_outliers.csv")

        # Klasörleri oluştur
        os.makedirs(reports_dir, exist_ok=True)

        # Veriyi yükle
        df = pd.read_csv(input_path, index_col='Date', parse_dates=True)

        # Z-Score hesapla
        from scipy import stats
        z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))

        # Aykırı değerleri işaretle (|Z| > 3)
        outliers = df[(z_scores > 3).any(axis=1)]

        # Raporu kaydet
        if not outliers.empty:
            outliers.to_csv(output_path)
            print(f"{symbol} aykırı değer raporu: {output_path}")
            return output_path
        else:
            print(f"{symbol} için aykırı değer bulunamadı")
            return None

    except Exception as e:
        print(f"{symbol} aykırı değer analiz hatası: {str(e)}")
        return None


# Örnek kullanım
if __name__ == "__main__":
    # Config dosyasını yükle
    config = load_config()
    symbols = config['symbols']
    processed_data_dir = config['paths']['processed_data']
    reports_dir = config['paths']['reports']

    # Tüm hisseleri analiz et
    for symbol in symbols:
        detect_outliers(
            symbol=symbol,
            processed_data_dir=processed_data_dir,
            reports_dir=reports_dir
        )