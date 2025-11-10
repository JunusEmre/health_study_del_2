import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple, List
import os

class HealthDataAnalyzer:
    """
    Klass för att utföra grundläggande och avancerade analyser
    på hälsostudiedata.
    """
    PLOT_DIR = 'plots'

    def __init__(self, filepath: str):
        """
        Initierar analysverktyget genom att läsa in datan.
        :param filepath: Sökvägen till CSV-filen.
        """
        self.filepath = filepath
        self.df = None
        self._load_data()
        np.random.seed(42)

    def _load_data(self):
        """ Laddar in data från CSV-filen. """
        try:
            self.df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Filen hittades inte på sökvägen: {self.filepath}")

    def data_cleaning(self) -> str:
        """
        Steg för att förbereda och validera datan.
        Kontrollerar efter dubbletter och saknade värden.
        :return: Sammanfattning av rengöringsresultatet.
        """
        if self.df is None:
            return "Databasen har inte laddats in."

        # 1. Kontrollera och hantera saknade värden (Missing Values)
        missing_count = self.df.isnull().sum().sum()
        if missing_count > 0:
            # I en verklig situation skulle man här välja att imputera eller ta bort rader.
            # För denna data vet vi att det inte finns några, men logiken inkluderas.
            pass

        # 2. Kontrollera och hantera dubbletter (Duplicates)
        duplicate_count = self.df.duplicated().sum()
        if duplicate_count > 0:
            self.df.drop_duplicates(inplace=True)

        # 3. Datatypkonvertering och standardisering
        # Konvertera 'sex' och 'smoker' till kategoriska för minneseffektivitet.
        self.df['sex'] = self.df['sex'].astype('category')
        self.df['smoker'] = self.df['smoker'].astype('category')
        
        # Säkerställ att 'disease' är en boolean/int för beräkningar
        self.df['disease'] = self.df['disease'].astype(int)

        result = (f"Rengöringsstatus:\n"
                  f"- Saknade värden hittades: {missing_count}\n"
                  f"- Dubbla rader hittades: {duplicate_count}\n"
                  f"- Datan är nu redo för analys.")
        return result
    
    def get_descriptive_stats(self, columns: List[str]) -> pd.DataFrame:
        """
        Beräknar medel, median, min och max för specificerade kolumner.
        """
        statistik = self.df[columns].agg(['mean', 'median', 'min', 'max']).T
        statistik.columns = ['Medel', 'Median', 'Min', 'Max']
        return statistik

    def generate_plots(self) -> str:
        """
        Skapar de 3 nödvändiga graferna och sparar dem som PNG.
        Visar dem också i notebooken.
        """
        if not os.path.exists(self.PLOT_DIR):
            os.makedirs(self.PLOT_DIR)

        # Grafik 1: Histogram över systoliskt blodtryck
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df['systolic_bp'], kde=True, bins=20, color='skyblue')
        plt.title('1. Fördelning av Systoliskt Blodtryck', fontsize=14)
        plt.xlabel('Systoliskt Blodtryck', fontsize=12)
        plt.ylabel('Frekvens (Antal)', fontsize=12)
        plt.grid(axis='y', alpha=0.5)
        save_path_1 = os.path.join(self.PLOT_DIR, 'graf_1_histogram_bp.png')
        plt.savefig(save_path_1)
        plt.show()

        # Grafik 2: Boxplot över vikt per kön
        plt.figure(figsize=(6, 8))
        sns.boxplot(x='sex', y='weight', data=self.df, hue='sex', palette={'M': 'orange', 'F': 'green'}, legend=False) 
        plt.title('2. Viktfördelning per Kön', fontsize=14)
        plt.xlabel('Kön', fontsize=12)
        plt.ylabel('Vikt (kg)', fontsize=12)
        save_path_1 = os.path.join(self.PLOT_DIR, 'graf_2_boxplot_vikt.png')
        plt.savefig(save_path_1)
        plt.show()

        # Grafik 3: Stapeldiagram över andelen rökare
        rokarandel = self.df['smoker'].value_counts(normalize=True) * 100
        plt.figure(figsize=(7, 5))
        rokarandel.plot(kind='bar', color=['darkred', 'gray'])
        plt.title('3. Andel Rökare vs Icke-Rökare', fontsize=14)
        plt.xlabel('Rökare (smoker)', fontsize=12)
        plt.ylabel('Andel (%)', fontsize=12)
        plt.xticks(rotation=0)
        for index, value in enumerate(rokarandel):
            plt.text(index, value + 0.5, f'{value:.2f}%', ha='center')
        plt.grid(axis='y', alpha=0.5)
        save_path_1 = os.path.join(self.PLOT_DIR, 'graf_3_stapeldiagram_rokarandel.png')
        plt.savefig(save_path_1)
        plt.show()
        
        return "Alla 3 grafer har skapats och sparats i PNG-filer samt visats i notebooken."
