"""Préparation des données Seattle 2016 – partagée entre le notebook et le dashboard.

Règles (retours de l'évaluateur P4 + consignes de Douglas) :
- aucune variable issue des relevés de consommation (EUI, kBtu, kWh, therms, intensités) n'entre dans les features ;
- seules les *proportions* des sources d'énergie (relevé de référence de 1re année, autorisé par Douglas) sont dérivées ;
- les cibles sont passées au log.
"""
import numpy as np
import pandas as pd

RESIDENTIAL_BT = ['Multifamily LR (1-4)', 'Multifamily MR (5-9)', 'Multifamily HR (10+)']
RESIDENTIAL_PT = ['Low-Rise Multifamily', 'Mid-Rise Multifamily', 'High-Rise Multifamily']
TARGETS = {'GHG': 'TotalGHGEmissions', 'Energy': 'SiteEnergyUse(kBtu)'}
TARGET_LABELS = {'GHG': 'Émissions de GES (t CO2e/an)', 'Energy': "Consommation d'énergie (kBtu/an)"}
# Centre-ville de Seattle (Westlake Center) pour une distance au centre
DOWNTOWN = (47.6115, -122.3374)

NUM = ['Latitude', 'Longitude', 'dist_downtown_km', 'BuildingAge', 'NumberofBuildings', 'NumberofFloors',
       'log_GFATotal', 'log_GFABuilding', 'parking_share', 'largest_use_share', 'second_use_share', 'third_use_share',
       'n_use_types', 'steam_share', 'elec_share', 'gas_share']
CAT = ['BuildingType', 'PrimaryPropertyType', 'LargestPropertyUseType', 'Neighborhood']
ESS = 'ENERGYSTARScore'


def load_and_filter(path):
    """Charge le CSV brut et garde les bâtiments non résidentiels exploitables. Renvoie (df, journal du filtrage)."""
    df = pd.read_csv(path)
    log = [('Relevés 2016 bruts', len(df))]
    df = df[~df.BuildingType.isin(RESIDENTIAL_BT) & ~df.PrimaryPropertyType.isin(RESIDENTIAL_PT)]
    log.append(('Hors habitation', len(df)))
    df = df[df.Outlier.isna() & (df.ComplianceStatus == 'Compliant')]
    log.append(('Conformes, non outliers', len(df)))
    df = df[(df['SiteEnergyUse(kBtu)'] > 0) & (df.TotalGHGEmissions > 0) & (df['PropertyGFABuilding(s)'] > 0)]
    log.append(('Cibles > 0', len(df)))
    df = df.set_index('OSEBuildingID')
    return df, log


def make_features(df):
    """Variables structurelles + proportions des sources d'énergie. N'utilise aucune quantité consommée en valeur absolue."""
    X = pd.DataFrame(index=df.index)
    X['Latitude'], X['Longitude'] = df.Latitude, df.Longitude
    # distance haversine au centre-ville
    lat1, lon1, lat2, lon2 = map(np.radians, [df.Latitude, df.Longitude, DOWNTOWN[0], DOWNTOWN[1]])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    X['dist_downtown_km'] = 6371 * 2 * np.arcsin(np.sqrt(a))
    X['BuildingAge'] = 2016 - df.YearBuilt
    X['NumberofBuildings'] = df.NumberofBuildings.fillna(1).clip(lower=1)
    X['NumberofFloors'] = df.NumberofFloors.clip(lower=1)
    gfa = df.PropertyGFATotal.clip(lower=1)
    X['log_GFATotal'] = np.log1p(df.PropertyGFATotal)
    X['log_GFABuilding'] = np.log1p(df['PropertyGFABuilding(s)'])
    X['parking_share'] = (df.PropertyGFAParking / gfa).clip(0, 1)
    X['largest_use_share'] = (df.LargestPropertyUseTypeGFA.fillna(df['PropertyGFABuilding(s)']) / gfa).clip(0, 1.5)
    X['second_use_share'] = (df.SecondLargestPropertyUseTypeGFA.fillna(0) / gfa).clip(0, 1.5)
    X['third_use_share'] = (df.ThirdLargestPropertyUseTypeGFA.fillna(0) / gfa).clip(0, 1.5)
    X['n_use_types'] = df.ListOfAllPropertyUseTypes.fillna('').str.split(',').apply(lambda l: len([u for u in l if u.strip()])).clip(lower=1)
    # Proportions des sources (relevé de référence de 1re année, autorisé) – pas les quantités
    tot = (df['SteamUse(kBtu)'].fillna(0) + df['Electricity(kBtu)'].fillna(0) + df['NaturalGas(kBtu)'].fillna(0)).replace(0, np.nan)
    X['steam_share'] = (df['SteamUse(kBtu)'].fillna(0) / tot).fillna(0).clip(0, 1)
    X['elec_share'] = (df['Electricity(kBtu)'].fillna(0) / tot).fillna(1).clip(0, 1)
    X['gas_share'] = (df['NaturalGas(kBtu)'].fillna(0) / tot).fillna(0).clip(0, 1)
    X['BuildingType'] = df.BuildingType.replace({'Nonresidential WA': 'NonResidential'})
    X['PrimaryPropertyType'] = df.PrimaryPropertyType.replace({'Office': 'Small- and Mid-Sized Office'})
    X['LargestPropertyUseType'] = df.LargestPropertyUseType.fillna(df.PrimaryPropertyType)
    X['Neighborhood'] = df.Neighborhood.str.upper().replace({'DELRIDGE NEIGHBORHOODS': 'DELRIDGE'})
    X[ESS] = df[ESS]
    return X


def targets(df):
    return {k: np.log(df[v].values) for k, v in TARGETS.items()}


class NNPreprocessor:
    """Prétraitement des réseaux (MLP/TabM), appris sur le train uniquement :
    - numériques : imputation médiane (+ indicateur de manque pour l'ENERGY STAR Score) puis QuantileTransformer -> loi normale ;
    - catégorielles : modalités rares (< min_count) et inconnues regroupées dans 'RARE', puis codes entiers.
    """
    def __init__(self, use_ess=False, min_count=10, seed=42):
        self.use_ess, self.min_count, self.seed = use_ess, min_count, seed

    def num_cols(self):
        return NUM + ([ESS, 'ESS_missing'] if self.use_ess else [])

    def _num(self, X):
        Z = X[NUM].copy()
        if self.use_ess:
            Z[ESS] = X[ESS]
            Z['ESS_missing'] = X[ESS].isna().astype(float)
        return Z

    def fit(self, X):
        from sklearn.preprocessing import QuantileTransformer
        Z = self._num(X)
        self.medians = Z.median()
        V = Z.fillna(self.medians).values.astype(float)
        self.qt = QuantileTransformer(n_quantiles=max(min(len(V) // 30, 1000), 10), output_distribution='normal',
                                      subsample=10**9, random_state=self.seed)
        noise = np.random.default_rng(self.seed).normal(0, 1e-5, V.shape) * V.std(0, keepdims=True)
        self.qt.fit(V + noise)
        self.categories = {}
        for c in CAT:
            vc = X[c].value_counts()
            self.categories[c] = sorted(vc[vc >= self.min_count].index.tolist()) + ['RARE']
        return self

    def cat_codes(self, X):
        out = []
        for c in CAT:
            cats = self.categories[c]
            v = X[c].where(X[c].isin(cats), 'RARE')
            out.append(pd.Categorical(v, categories=cats).codes)
        return np.stack(out, 1).astype(np.int64)

    def transform(self, X):
        x_num = self.qt.transform(self._num(X).fillna(self.medians).values.astype(float)).astype(np.float32)
        return x_num, self.cat_codes(X)

    @property
    def cardinalities(self):
        return [len(self.categories[c]) for c in CAT]
