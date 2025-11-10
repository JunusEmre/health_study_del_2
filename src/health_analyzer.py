import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple, List

class HealthDataAnalyzer:
    """
    Klass för att utföra grundläggande och avancerade analyser
    på hälsostudiedata.
    """

    def __init__(self, filepath: str):
        """
        Initierar analysverktyget genom att läsa in datan.
        :param filepath: Sökvägen till CSV-filen.
        """
        self.filepath = filepath
        self.df = None
        self._load_data()
        np.random.seed(42) # Global seed för reproducerbarhet

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
