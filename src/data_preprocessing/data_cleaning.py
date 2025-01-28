import pandas as pd
import os
import yaml


def load_config():
    """Config.yaml dosyasını yükler."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def clean_data(symbol: str, raw_data_dir: str, processed_data_dir: str) -> str:
    """
    Hisse sembolüne özel veri temizleme işlemi yapar

    Parameters:
    - symbol (str): Hisse sembolü (Örnek: "GARAN.IS")
    - raw_data_dir (str): Ham veri klasör yolu
    - processed_data_dir (str): İşlenmiş veri klasör yolu

    Returns:
    str: İşlenmiş veri dosya yolu
    """
    try:
        # Dosya yollarını oluştur
        input_path = os.path.join(raw_data_dir, f"{symbol}_historical_data.csv")
        output_path = os.path.join(processed_data_dir, f"{symbol}_processed.csv")

        # Klasörleri oluştur
        os.makedirs(processed_data_dir, exist_ok=True)

        # Veriyi yükle
        df = pd.read_csv(input_path, parse_dates=['Date'], index_col='Date')

        # 1. Eksik veri işlemleri
        df.ffill(inplace=True)  # Önceki değerlerle doldur
        df.bfill(inplace=True)  # Sonraki değerlerle doldur

        # 2. Gereksiz sütunları kaldır
        columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[columns_to_keep]

        # 3. Anomali kontrolü
        df = df[(df['Volume'] > 0) & (df['Close'] > 0)]

        # 4. Veriyi kaydet
        df.to_csv(output_path, index=True)
        print(f"{symbol} verisi başarıyla temizlendi: {output_path}")
        return output_path

    except Exception as e:
        print(f"{symbol} temizleme hatası: {str(e)}")
        return None


# Örnek kullanım
if __name__ == "__main__":
    # Config dosyasını yükle
    config = load_config()
    symbols = config['symbols']
    raw_data_dir = config['paths']['raw_data']
    processed_data_dir = config['paths']['processed_data']

    # Tüm hisseleri temizle
    for symbol in symbols:
        clean_data(
            symbol=symbol,
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir
        )