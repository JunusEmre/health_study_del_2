import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple, List
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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
    
    # --- NY METOD FÖR DEL 2 (Datapreprocessing) ---

    def data_preprocessing(self) -> str:
        """
        Förbereder datan för maskininlärning genom standardisering och one-hot encoding.
        Detta är ett viktigt steg i analys-pipelinen.
        """
        df_model = self.df.copy()

        # 1. One-Hot Encoding av kategoriska variabler
        df_model = pd.get_dummies(df_model, columns=['sex', 'smoker'], drop_first=True)
        df_model.drop(columns=['id'], inplace=True)
        df_model.rename(columns={'sex_M': 'sex_Male', 'smoker_Yes': 'smoker_Yes'}, inplace=True)
        
        # 2. Standardisering (Scaling) av numeriska variabler
        numeric_cols = ['age', 'height', 'weight', 'systolic_bp', 'cholesterol']
        
        scaler = StandardScaler()
        df_model[numeric_cols] = scaler.fit_transform(df_model[numeric_cols])
        
        # Lagra den bearbetade datan som ett nytt attribut
        self.df_processed = df_model
        
        return "Data standardiserad och kodad (One-Hot Encoded) för modellering i self.df_processed."


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
    def run_simulation(self, sample_size: int = 1000) -> Dict[str, float]:
        """
        Simulerar slumpade personer baserat på datasetets sjukdomssannolikhet.
        """
        verklig_andel = self.df['disease'].mean()
        
        # Simulera med samma sannolikhet för sjukdom
        simulerade_resultat = np.random.choice(
            [0, 1],
            size=sample_size,
            p=[1 - verklig_andel, verklig_andel]
        )
        
        simulerad_andel = np.mean(simulerade_resultat)
        
        return {
            "verklig_andel": verklig_andel * 100,
            "simulerad_andel": simulerad_andel * 100,
            "sample_size": sample_size
        }

    def calculate_ci(self, method: str = 'normal', conf_level: float = 0.95) -> Dict[str, float]:
        """
        Beräknar konfidensintervallet för medelvärdet av systolic_bp.
        :param method: 'normal' (Normalapproximation) eller 'bootstrap'.
        """
        data_bp = self.df['systolic_bp'].values
        
        if method == 'normal':
            medelvarde = data_bp.mean()
            standardavvikelse = data_bp.std()
            standardfel = standardavvikelse / np.sqrt(len(data_bp))
            
            # Använd Scipy för Z-baserat konfidensintervall (för stort N)
            nedre, ovre = stats.norm.interval(
                conf_level,
                loc=medelvarde,
                scale=standardfel
            )
            return {"nedre_grans": nedre, "ovre_grans": ovre, "metod": "Normalapproximation", "medel": medelvarde}
            
        elif method == 'bootstrap':
            antal_aterupprepningar = 10000 
            bootstrap_medel = []
            
            for _ in range(antal_aterupprepningar):
                bootstrap_sample = np.random.choice(data_bp, size=len(data_bp), replace=True)
                bootstrap_medel.append(np.mean(bootstrap_sample))
            
            medelvarde = data_bp.mean()
            
            # Beräkna percentiler för 95% CI
            nedre = np.percentile(bootstrap_medel, (1 - conf_level) / 2 * 100)
            ovre = np.percentile(bootstrap_medel, (1 + conf_level) / 2 * 100)
            
            return {"nedre_grans": nedre, "ovre_grans": ovre, "metod": "Bootstrap", "medel": medelvarde}
            
        else:
            raise ValueError("Metoden måste vara 'normal' eller 'bootstrap'.")
        
    def run_hypothesis_test(self, alpha: float = 0.05) -> Dict:
        """
        Testar hypotesen: "Rökare har högre medel-blodtryck än icke-rökare."
        Använder oberoende t-test (Welch's).
        """
        roakare_bp = self.df[self.df['smoker'] == 'Yes']['systolic_bp']
        icke_roakare_bp = self.df[self.df['smoker'] == 'No']['systolic_bp']
        
        medel_roakare = roakare_bp.mean()
        medel_icke_roakare = icke_roakare_bp.mean()
        
        # T-test, ensidigt (alternative='greater') och Welch's (equal_var=False)
        t_stat, p_value = stats.ttest_ind(
            roakare_bp,
            icke_roakare_bp,
            equal_var=False,
            alternative='greater'
        )
        
        # Sammanfatta resultat
        slutsats = "Förkasta nollhypotesen (H0)." if p_value < alpha else "Behåll nollhypotesen (H0)."
        
        return {
            "medel_roakare": medel_roakare,
            "medel_icke_roakare": medel_icke_roakare,
            "t_stat": t_stat,
            "p_value": p_value,
            "alpha": alpha,
            "slutsats": slutsats
        }
    
    def run_power_simulation(self, iterations: int = 1000, alpha: float = 0.05) -> float:
        """
        Simulerar testets statistiska styrka (Power) baserat på observerade parametrar.
        """
        resultat_test = self.run_hypothesis_test(alpha=alpha)
        
        n_roakare = len(self.df[self.df['smoker'] == 'Yes'])
        n_icke_roakare = len(self.df[self.df['smoker'] == 'No'])
        std_roakare = self.df[self.df['smoker'] == 'Yes']['systolic_bp'].std()
        std_icke_roakare = self.df[self.df['smoker'] == 'No']['systolic_bp'].std()

        medel_icke_roakare_sanning = resultat_test['medel_icke_roakare']
        medel_roakare_sanning = resultat_test['medel_roakare']
        
        antal_avslag = 0 

        for _ in range(iterations):
            # Skapa simulerad data med den antagna sanna skillnaden
            sim_roakare_bp = np.random.normal(loc=medel_roakare_sanning, scale=std_roakare, size=n_roakare)
            sim_icke_roakare_bp = np.random.normal(loc=medel_icke_roakare_sanning, scale=std_icke_roakare, size=n_icke_roakare)
            
            # Utför t-testet (ensidigt: 'greater')
            t_stat_sim, p_value_sim = stats.ttest_ind(
                sim_roakare_bp,
                sim_icke_roakare_bp,
                equal_var=False,
                alternative='greater'
            )
            
            if p_value_sim < alpha:
                antal_avslag += 1

        power = antal_avslag / iterations
        return power