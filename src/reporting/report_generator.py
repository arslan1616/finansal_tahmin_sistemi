# src/reporting/report_generator.py

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from fpdf import FPDF


class ReportGenerator:
    """
    Tahmin sonuçları için otomatik rapor oluşturur.
    """

    def __init__(self, stock_data, predictions, model_performance):
        """
        Args:
            stock_data: Hisse senedi verileri
            predictions: Model tahminleri
            model_performance: Model performans metrikleri
        """
        self.stock_data = stock_data
        self.predictions = predictions
        self.model_performance = model_performance
        self.report_date = datetime.now().strftime("%Y-%m-%d")

    def create_performance_summary(self):
        """Model performans özetini oluşturur."""
        summary = pd.DataFrame({
            'Metrik': ['MSE', 'RMSE', 'MAE', 'R2'],
            'Değer': [
                self.model_performance['mse'],
                self.model_performance['rmse'],
                self.model_performance['mae'],
                self.model_performance['r2']
            ]
        })
        return summary

    def plot_predictions(self):
        """Tahmin grafiğini oluşturur."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.stock_data.index, self.stock_data['Close'], label='Gerçek Değer')
        plt.plot(self.predictions.index, self.predictions['Predicted'], label='Tahmin')
        plt.title('Hisse Senedi Fiyat Tahmini')
        plt.xlabel('Tarih')
        plt.ylabel('Fiyat (TL)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Grafiği geçici dosyaya kaydet
        plt.savefig('temp_prediction_plot.png')
        plt.close()

    def generate_pdf_report(self):
        """PDF raporu oluşturur."""
        pdf = FPDF()
        pdf.add_page()

        # Başlık
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Finansal Tahmin Raporu', ln=True, align='C')
        pdf.cell(0, 10, f'Tarih: {self.report_date}', ln=True, align='C')

        # Performans özeti
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Model Performans Özeti:', ln=True)

        summary = self.create_performance_summary()
        pdf.set_font('Arial', '', 12)
        for _, row in summary.iterrows():
            pdf.cell(0, 10, f'{row["Metrik"]}: {row["Değer"]:.4f}', ln=True)

        # Tahmin grafiği
        self.plot_predictions()
        pdf.image('temp_prediction_plot.png', x=10, y=None, w=190)

        # Geçici dosyayı sil
        os.remove('temp_prediction_plot.png')

        # Raporu kaydet
        report_path = f'reports/financial_prediction_report_{self.report_date}.pdf'
        os.makedirs('reports', exist_ok=True)
        pdf.output(report_path)
        print(f"Rapor oluşturuldu: {report_path}")

    def generate_excel_report(self):
        """Excel raporu oluşturur."""
        report_data = pd.DataFrame({
            'Tarih': self.predictions.index,
            'Gerçek Değer': self.stock_data['Close'],
            'Tahmin': self.predictions['Predicted'],
            'Fark': self.stock_data['Close'] - self.predictions['Predicted']
        })

        report_path = f'reports/financial_prediction_report_{self.report_date}.xlsx'
        os.makedirs('reports', exist_ok=True)

        with pd.ExcelWriter(report_path) as writer:
            report_data.to_excel(writer, sheet_name='Tahminler')
            self.create_performance_summary().to_excel(writer, sheet_name='Performans Özeti')

        print(f"Excel raporu oluşturuldu: {report_path}")