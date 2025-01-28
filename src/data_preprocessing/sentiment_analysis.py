from transformers import pipeline

def analyze_sentiment(news):
    """
    Piyasa haberlerinin sentiment analizini yapar.

    Parameters:
    - news (list): Analiz edilecek haber metinleri listesi.
    """
    # Sentiment analizi modelini yükle
    sentiment_analyzer = pipeline("sentiment-analysis")

    # Sentiment analizi yap
    results = sentiment_analyzer(news)
    for i, result in enumerate(results):
        print(f"Haber {i+1}: {result['label']} (Skor: {result['score']:.2f})")

# Örnek kullanım
if __name__ == "__main__":
    news = [
        "Garanti Bankası, yılın ilk çeyreğinde rekor kâr açıkladı.",
        "Piyasalar, küresel ekonomik belirsizlikler nedeniyle düşüşte."
    ]
    analyze_sentiment(news)