import pandas as pd
import os

# Proje kök dizinini al
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Veri yolları
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# İşlenmiş veriler için klasörü oluştur
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def process_stock_data(file_path):
    """
    Belirtilen hisse senedi dosyasını işler ve kaydeder.

    Parameters:
    - file_path (str): Hisse verisinin bulunduğu dosyanın yolu.
    """
    try:
        # Dosya adından hisse kodunu çıkar
        filename = os.path.basename(file_path)
        stock_symbol = filename.split("_")[0]

        # Veriyi oku
        data = pd.read_csv(file_path, skiprows=2)

        # Sütun isimlerini düzelt
        data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

        # Tarih sütununu datetime formatına çevir
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

        # İşlenmiş veriyi kaydet
        processed_file_path = os.path.join(PROCESSED_DATA_DIR, f"{stock_symbol}_processed_data.csv")
        data.to_csv(processed_file_path, index=False)
        print(f"{stock_symbol} verisi işlendi ve kaydedildi: {processed_file_path}")

    except Exception as e:
        print(f"{stock_symbol} veri analizi sırasında hata oluştu: {e}")

if __name__ == "__main__":
    # data/raw içindeki tüm CSV dosyalarını işle
    for file in os.listdir(RAW_DATA_DIR):
        if file.endswith(".csv"):
            file_path = os.path.join(RAW_DATA_DIR, file)
            process_stock_data(file_path)