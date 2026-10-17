"""Dashboard de la preuve de concept : prédiction des émissions de GES des bâtiments non résidentiels de Seattle avec TabM
(ICLR 2025), comparé au Random Forest de mon projet P3 (baseline).

Accessibilité (WCAG 2.1) prise en compte :
- 1.1.1 Contenu non textuel : chaque graphique a un titre, une description textuelle et ses données en tableau (expander) ;
- 1.4.1 Utilisation de la couleur : palette Okabe-Ito / cividis adaptée aux daltoniens, couleur doublée par des formes, motifs ou libellés ;
- 1.4.3 / 1.4.11 Contraste : texte foncé sur fond clair (ratio > 4,5:1), éléments graphiques contrastés (> 3:1) ;
- 1.4.4 / 1.4.10 Redimensionnement : mise en page fluide, graphiques responsives ;
- 2.1.1 Clavier : uniquement des composants Streamlit natifs, utilisables au clavier ;
- 3.3.2 Étiquettes : chaque champ de saisie a un libellé explicite et une aide.
"""
import json
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from seattle_prep import CAT, DOWNTOWN, NUM

ART = Path(__file__).parent / 'artifacts'
OI = {'bleu': '#0072B2', 'orange': '#E69F00', 'vert': '#009E73', 'rose': '#CC79A7', 'ciel': '#56B4E9', 'vermillon': '#D55E00', 'noir': '#000000'}
MODELS = ['Random Forest', 'TabM']
MODEL_COLORS = {'Random Forest': OI['orange'], 'TabM': OI['bleu']}
MODEL_PATTERNS = {'Random Forest': '/', 'TabM': ''}
FEATS = NUM + CAT
TARGET = 'TotalGHGEmissions'
LABELS = {
    'Latitude': 'Latitude', 'Longitude': 'Longitude', 'dist_downtown_km': 'Distance au centre-ville (km)', 'BuildingAge': 'Âge du bâtiment (ans)',
    'NumberofBuildings': 'Nombre de bâtiments', 'NumberofFloors': "Nombre d'étages", 'log_GFATotal': 'log surface totale (pi²)',
    'log_GFABuilding': 'log surface bâtie (pi²)', 'parking_share': 'Part de parking', 'largest_use_share': "Part de l'usage principal",
    'second_use_share': "Part du 2e usage", 'third_use_share': "Part du 3e usage", 'n_use_types': "Nombre d'usages",
    'steam_share': 'Part vapeur (steam)', 'elec_share': 'Part électricité', 'gas_share': 'Part gaz naturel',
    'BuildingType': 'Type de bâtiment', 'PrimaryPropertyType': 'Type principal', 'LargestPropertyUseType': 'Usage principal', 'Neighborhood': 'Quartier',
    TARGET: 'Émissions GES (t CO2e/an)',
}
PLOT_LAYOUT = dict(template='simple_white', font=dict(size=14, color='#1F232B'), margin=dict(l=10, r=10, t=60, b=10), title_font=dict(size=17))

st.set_page_config(page_title='Seattle – PoC TabM', page_icon='🏢', layout='wide')
st.markdown("""<style>
html, body, [class*="css"] { font-size: 16px; }
:focus-visible { outline: 3px solid #0072B2 !important; outline-offset: 2px; }
.small-note { color: #3B4250; font-size: 0.92rem; }
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ données & modèle
@st.cache_data
def load_data():
    b = pd.read_csv(ART / 'buildings.csv', index_col=0)
    preds = pd.read_csv(ART / 'test_predictions.csv', index_col=0)
    res = json.load(open(ART / 'results.json'))
    bg = pd.read_csv(ART / 'shap_background.csv')
    return b, preds, res, bg


@st.cache_resource
def load_model():
    bundle = joblib.load(ART / 'tabm_preprocessing.joblib')
    sess = ort.InferenceSession(str(ART / 'tabm_GHG.onnx'), providers=['CPUExecutionProvider'])
    return bundle, sess


B, PREDS, RES, BG = load_data()
BUNDLE, SESS = load_model()
CAT_VALUES = {c: sorted(B[c].dropna().unique()) for c in CAT}


def predict_log(X):
    """X : DataFrame aux colonnes FEATS (catégorielles en texte). Renvoie log(émissions prédites)."""
    xn, xc = BUNDLE['prep'].transform(X)
    return SESS.run(None, {'x_num': xn, 'x_cat': xc})[0][:, 0] * BUNDLE['sd'] + BUNDLE['mu']


def encode(X):
    Z = X[FEATS].copy()
    for c in CAT:
        Z[c] = pd.Categorical(X[c], categories=CAT_VALUES[c]).codes
    return Z.astype(float)


def decode(Z):
    X = pd.DataFrame(np.asarray(Z), columns=FEATS)
    for c in CAT:
        X[c] = np.array(CAT_VALUES[c])[np.clip(X[c].astype(int).values, 0, len(CAT_VALUES[c]) - 1)]
    return X


@st.cache_data(show_spinner=False)
def local_shap(row_json):
    import shap
    row = pd.read_json(StringIO(row_json), orient='split')
    bg = BG.copy()
    for c in CAT:  # le fond de référence a été encodé avec les mêmes listes triées de modalités
        bg[c] = bg[c].clip(0, len(CAT_VALUES[c]) - 1)
    ex = shap.PermutationExplainer(lambda Z: predict_log(decode(Z)), shap.maskers.Independent(bg, max_samples=100))
    sv = ex(encode(row), max_evals=2 * len(FEATS) + 1, silent=True)
    return sv.values[0], float(sv.base_values[0])


def described_chart(fig, description, data=None, key=None):
    # contraste WCAG : graduations et titres d'axes en gris foncé (> 4,5:1 sur blanc)
    fig.update_xaxes(tickfont=dict(color='#1F232B', size=13), title_font=dict(color='#1F232B'))
    fig.update_yaxes(tickfont=dict(color='#1F232B', size=13), title_font=dict(color='#1F232B'))
    fig.update_traces(cliponaxis=False, selector=dict(type='bar'))
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.caption(f'Description du graphique : {description}')
    if data is not None:
        with st.expander('Voir les données du graphique (alternative textuelle)'):
            st.dataframe(data, use_container_width=True)


# ------------------------------------------------------------------ navigation
st.sidebar.title('PoC TabM – Seattle 2016')
page = st.sidebar.radio('Aller à la section', ['1. Exploration des données', '2. Prédiction', '3. Comparaison des modèles'])
st.sidebar.markdown('---')
st.sidebar.markdown("**Cible** : émissions de gaz à effet de serre des bâtiments **non résidentiels**, prédites à partir de "
                    "données **structurelles** uniquement (sans relevé de consommation).")
st.sidebar.markdown('**Modèle** : TabM (Gorishniy et al., ICLR 2025) – [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)')

# ================================================================== 1. EDA
if page.startswith('1'):
    st.title('Exploration des données – bâtiments non résidentiels de Seattle (2016)')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Bâtiments', f'{len(B):,}'.replace(',', ' '))
    c2.metric('Variables explicatives', len(FEATS))
    c3.metric('Types de propriété', B.PrimaryPropertyType.nunique())
    c4.metric('Quartiers', B.Neighborhood.nunique())
    with st.expander('Étapes de filtrage du jeu brut'):
        st.table(pd.DataFrame(RES['filter_log'], columns=['Étape', 'Bâtiments']))

    types = st.multiselect('Filtrer par type principal de propriété', CAT_VALUES['PrimaryPropertyType'], default=[], placeholder='Tous les types',
                           help="Laisser vide pour afficher tous les types.")
    D = B[B.PrimaryPropertyType.isin(types)] if types else B

    st.subheader('Statistiques descriptives')
    cols = [TARGET, 'BuildingAge', 'NumberofFloors', 'log_GFABuilding', 'elec_share', 'gas_share', 'steam_share']
    desc = D[cols].describe().T.rename(index=LABELS).rename(columns={'count': 'effectif', 'mean': 'moyenne', 'std': 'écart-type', '50%': 'médiane'})
    st.dataframe(desc.style.format('{:,.2f}'), use_container_width=True)

    st.subheader('Comptages par catégorie')
    var = st.selectbox('Variable catégorielle à compter', CAT, format_func=lambda c: LABELS[c])
    cnt = D[var].value_counts().rename_axis(LABELS[var]).reset_index(name='Bâtiments')
    fig = px.bar(cnt.head(20).iloc[::-1], x='Bâtiments', y=LABELS[var], orientation='h', text='Bâtiments', title=f'Nombre de bâtiments par {LABELS[var].lower()} (20 premiers)')
    fig.update_traces(marker_color=OI['bleu'], textposition='outside'); fig.update_layout(**PLOT_LAYOUT, height=520)
    described_chart(fig, f"Barres horizontales du nombre de bâtiments par {LABELS[var].lower()}. Modalité la plus fréquente : "
                         f"{cnt.iloc[0, 0]} ({cnt.iloc[0, 1]} bâtiments).", cnt, key='cnt')

    st.subheader('Distribution des émissions')
    logscale = st.checkbox("Échelle logarithmique (recommandée : distribution très asymétrique)", value=True)
    vals = np.log10(D[TARGET]) if logscale else D[TARGET]
    fig = px.histogram(x=vals, nbins=50, title="Distribution des émissions de GES" + (' (log10)' if logscale else ''),
                       labels={'x': ('log10 ' if logscale else '') + LABELS[TARGET], 'count': 'Bâtiments'})
    fig.update_traces(marker_color=OI['bleu'], marker_line_color='white', marker_line_width=1); fig.update_layout(**PLOT_LAYOUT, yaxis_title='Bâtiments')
    described_chart(fig, f"Histogramme des émissions. Médiane : {D[TARGET].median():,.0f} t ; moyenne : {D[TARGET].mean():,.0f} t – la moyenne bien "
                         "supérieure à la médiane traduit une forte asymétrie, d'où le passage au log pour la modélisation.", key='hist')

    st.subheader('Relation surface – émissions selon le type')
    top = D.PrimaryPropertyType.value_counts().index[:6]
    E = D[D.PrimaryPropertyType.isin(top)]
    fig = px.scatter(E, x='log_GFABuilding', y=np.log10(E[TARGET]), color='PrimaryPropertyType', symbol='PrimaryPropertyType',
                     color_discrete_sequence=list(OI.values()), hover_name='PropertyName',
                     labels={'log_GFABuilding': 'log surface bâtie (pi²)', 'y': 'log10 émissions GES (t CO2e)', 'PrimaryPropertyType': 'Type'},
                     title='Émissions vs surface (6 types les plus fréquents) – couleur et forme par type')
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white'))); fig.update_layout(**PLOT_LAYOUT, height=520)
    rho = np.corrcoef(D.log_GFABuilding, np.log10(D[TARGET]))[0, 1]
    described_chart(fig, f"Nuage de points : plus la surface est grande, plus les émissions sont élevées (corrélation des logs r = {rho:.2f}). "
                         "Chaque type a une couleur ET une forme de marqueur distinctes.", key='scatter')

    st.subheader('Carte des bâtiments')
    fig = px.scatter_map(D, lat='Latitude', lon='Longitude', color=np.log10(D[TARGET]), color_continuous_scale='cividis',
                         hover_name='PropertyName', hover_data={'PrimaryPropertyType': True, TARGET: ':.1f'},
                         zoom=10.3, height=560, map_style='open-street-map', labels={'color': 'log10 GES'},
                         title='Localisation des bâtiments – couleur = log10 des émissions')
    fig.update_layout(**PLOT_LAYOUT)
    described_chart(fig, f"Carte interactive de {len(D)} bâtiments. Les plus forts émetteurs se concentrent au centre-ville (Downtown) "
                         "et dans la zone industrielle de Greater Duwamish.", D[['PropertyName', 'Neighborhood', TARGET]].sort_values(TARGET, ascending=False).head(30), key='map')

    st.subheader('Corrélations (variables quantitatives ayant un sens)')
    cc = ['BuildingAge', 'NumberofFloors', 'log_GFABuilding', 'parking_share', 'n_use_types', 'dist_downtown_km', 'steam_share', 'elec_share', 'gas_share']
    C = D[cc].assign(**{'log GES': np.log(D[TARGET])}).corr('spearman').round(2)
    C.index = C.columns = [LABELS.get(c, c) for c in C.columns]
    fig = px.imshow(C, text_auto=True, color_continuous_scale='cividis', zmin=-1, zmax=1, aspect='auto', title='Corrélations de Spearman')
    fig.update_layout(**PLOT_LAYOUT, height=600)
    described_chart(fig, "Matrice de corrélation avec valeurs affichées dans chaque case. La surface bâtie est la variable la plus corrélée aux émissions ; "
                         "la part de gaz leur est positivement liée (le gaz émet plus que l'électricité, majoritairement hydroélectrique à Seattle).", C, key='corr')

# ================================================================== 2. Prédiction
elif page.startswith('2'):
    st.title('Prédiction des émissions de GES avec TabM')
    mode = st.radio("Données d'entrée", ['Choisir un bâtiment du jeu de test', 'Saisir les caractéristiques d’un bâtiment'], horizontal=True)
    test = B[B.split == 'test']
    if mode.startswith('Choisir'):
        nm = test.PropertyName.str.strip()
        nm = nm.where(nm != nm.str.upper(), nm.str.title())   # noms tout en majuscules → casse titre
        names = (nm + ' – ' + test.PrimaryPropertyType + ' (#' + test.index.astype(str) + ')').sort_values()
        choice = st.selectbox(f'Bâtiment ({len(test)} bâtiments jamais vus pendant l’entraînement)', names.index, format_func=lambda i: names[i])
        row = B.loc[[choice], FEATS]
        truth = B.loc[choice, TARGET]
        st.markdown(f"**Adresse** : {B.loc[choice, 'Address']} – **construit en** {int(B.loc[choice, 'YearBuilt'])} – **quartier** {B.loc[choice, 'Neighborhood']}")
    else:
        with st.form('saisie'):
            st.markdown('Renseignez les caractéristiques **structurelles** du bâtiment.')
            a, b, c = st.columns(3)
            ptype = a.selectbox('Type principal de propriété', CAT_VALUES['PrimaryPropertyType'], index=CAT_VALUES['PrimaryPropertyType'].index('Small- and Mid-Sized Office'))
            btype = b.selectbox('Type de bâtiment', CAT_VALUES['BuildingType'], index=CAT_VALUES['BuildingType'].index('NonResidential'))
            use = c.selectbox('Usage principal', CAT_VALUES['LargestPropertyUseType'], index=CAT_VALUES['LargestPropertyUseType'].index('Office') if 'Office' in CAT_VALUES['LargestPropertyUseType'] else 0)
            hood = a.selectbox('Quartier', CAT_VALUES['Neighborhood'], index=CAT_VALUES['Neighborhood'].index('DOWNTOWN'))
            year = b.number_input('Année de construction', 1900, 2016, 1965, help='Entre 1900 et 2016')
            floors = c.number_input("Nombre d'étages", 1, 99, 3)
            gfa = a.number_input('Surface bâtie (pieds carrés)', 1000, 10_000_000, 50_000, step=1000, help='1 m² ≈ 10,76 pi²')
            parking = b.slider('Part de la surface en parking', 0.0, 0.9, 0.0, 0.05)
            nbld = c.number_input('Nombre de bâtiments sur la parcelle', 1, 50, 1)
            elec = a.slider('Part électricité (relevé de référence)', 0.0, 1.0, 0.7, 0.05)
            steam = b.slider('Part vapeur (steam)', 0.0, 1.0, 0.0, 0.05)
            n_use = c.number_input("Nombre d'usages", 1, 13, 1)
            lat = a.number_input('Latitude', 47.49, 47.74, float(B[B.Neighborhood == 'DOWNTOWN'].Latitude.median()), format='%.4f')
            lon = b.number_input('Longitude', -122.42, -122.25, float(B[B.Neighborhood == 'DOWNTOWN'].Longitude.median()), format='%.4f')
            st.form_submit_button('Prédire')
        gas = max(0.0, 1 - elec - steam)
        if elec + steam > 1:
            st.warning('La somme électricité + vapeur dépasse 100 % : la part de gaz est fixée à 0.')
        dist = 6371 * 2 * np.arcsin(np.sqrt(np.sin(np.radians(DOWNTOWN[0] - lat) / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(DOWNTOWN[0])) * np.sin(np.radians(DOWNTOWN[1] - lon) / 2) ** 2))
        total = gfa / max(1e-6, 1 - parking)
        row = pd.DataFrame([{'Latitude': lat, 'Longitude': lon, 'dist_downtown_km': dist, 'BuildingAge': 2016 - year, 'NumberofBuildings': nbld,
                             'NumberofFloors': floors, 'log_GFATotal': np.log1p(total), 'log_GFABuilding': np.log1p(gfa), 'parking_share': parking,
                             'largest_use_share': (1 - parking) if n_use == 1 else 0.7 * (1 - parking), 'second_use_share': 0.0 if n_use == 1 else 0.25 * (1 - parking),
                             'third_use_share': 0.0, 'n_use_types': n_use, 'steam_share': steam, 'elec_share': elec, 'gas_share': gas,
                             'BuildingType': btype, 'PrimaryPropertyType': ptype, 'LargestPropertyUseType': use, 'Neighborhood': hood}])
        truth = None

    ghg = float(np.exp(predict_log(row))[0])
    tabm_test = [r for r in RES['test'] if r['modèle'] == 'TabM'][0]
    st.subheader('Résultat de la prédiction (TabM)')
    st.metric('Émissions de GES prédites', f'{ghg:,.1f} t CO2e/an'.replace(',', ' '),
              delta=None if truth is None else f"réel : {truth:,.1f} t".replace(',', ' '), delta_color='off')
    st.markdown(f"<p class='small-note'>Ordre de grandeur de l'incertitude : l'erreur relative médiane de TabM sur le jeu de test est d'environ "
                f"{tabm_test['MedAPE_%']:.0f} %.</p>", unsafe_allow_html=True)

    if truth is not None:
        cmp = pd.DataFrame({'Modèle': MODELS, 'Prédiction GES (t)': [PREDS.loc[choice, f'pred_{m}'] for m in MODELS]})
        cmp['Erreur relative'] = cmp['Prédiction GES (t)'] / truth - 1
        fig = go.Figure()
        for _, r in cmp.iterrows():
            fig.add_bar(x=[r['Modèle']], y=[r['Prédiction GES (t)']], name=r['Modèle'], marker_color=MODEL_COLORS[r['Modèle']],
                        marker_pattern_shape=MODEL_PATTERNS[r['Modèle']], marker_line_color='black', marker_line_width=1,
                        text=[f"{r['Prédiction GES (t)']:,.1f} t ({r['Erreur relative']:+.0%})"], textposition='outside')
        fig.add_hline(y=truth, line_dash='dash', line_color='black', annotation_text=f"Valeur réelle mesurée : {truth:,.1f} t",
                      annotation_position='top right', annotation_bgcolor='white')
        fig.update_layout(**PLOT_LAYOUT, title='Baseline (Random Forest) et TabM sur ce bâtiment', showlegend=False, yaxis_title='t CO2e/an', height=430)
        described_chart(fig, 'Barres des prédictions de chaque modèle (hachures = Random Forest, plein = TabM), ligne pointillée = valeur réelle mesurée.', cmp.round(3), key='cmp')

    st.subheader('Pourquoi cette prédiction ? (explication locale SHAP)')
    with st.spinner('Calcul des valeurs de Shapley…'):
        vals, base = local_shap(row[FEATS].to_json(orient='split'))
    E = pd.DataFrame({'variable': [LABELS[f] for f in FEATS], 'valeur': [row[f].iloc[0] for f in FEATS], 'shap': vals})
    E['abs'] = E.shap.abs(); E = E.sort_values('abs', ascending=False).head(10).iloc[::-1]
    E['effet'] = np.where(E.shap > 0, 'augmente', 'diminue')
    E['libellé'] = E.apply(lambda r: f"{r['variable']} = {r['valeur']:.2f}" if isinstance(r['valeur'], (int, float, np.floating)) else f"{r['variable']} = {r['valeur']}", axis=1)
    fig = go.Figure(go.Bar(x=E.shap, y=E['libellé'], orientation='h', marker_color=np.where(E.shap > 0, OI['vermillon'], OI['bleu']),
                           marker_pattern_shape=np.where(E.shap > 0, '', '/'), marker_line_color='black', marker_line_width=0.8,
                           text=[f"{'+' if v > 0 else '−'}{abs(v):.2f}" for v in E.shap], textposition='outside',
                           hovertext=[f'{e} les émissions' for e in E.effet]))
    fig.update_layout(**PLOT_LAYOUT, title=f'Contributions au log des émissions (base = {np.exp(base):,.0f} t, prédiction = {ghg:,.1f} t)',
                      xaxis_title='Contribution SHAP (log GES) – à droite : augmente les émissions', height=470)
    fig.add_vline(x=0, line_color='black')
    span = float(E['abs'].max()) * 1.35
    fig.update_xaxes(range=[-span if (E.shap < 0).any() else -0.05 * span, span if (E.shap > 0).any() else 0.05 * span])
    top = E.iloc[::-1].iloc[0]
    described_chart(fig, f"Les 10 variables qui pèsent le plus sur cette prédiction. Barres vermillon pleines = augmentent les émissions, "
                         f"barres bleues hachurées = les diminuent. Facteur principal : {top['libellé']} ({top['effet']} les émissions).",
                    E[['libellé', 'shap', 'effet']].iloc[::-1].round(3), key='shap')
    st.markdown("<p class='small-note'>Une contribution de +0,69 en log correspond à un facteur ×2 sur les émissions.</p>", unsafe_allow_html=True)

# ================================================================== 3. Comparaison
else:
    st.title('Comparaison : TabM vs Random Forest (baseline du P3)')
    T = pd.DataFrame(RES['test']).set_index('modèle').loc[MODELS]; S = pd.DataFrame(RES['test_std']).set_index('modèle').loc[MODELS]
    CV = pd.DataFrame(RES['cv']).set_index('modèle').loc[MODELS]
    metric = st.selectbox('Métrique (jeu de test, 5 graines)', ['RMSE_log', 'R2_log', 'MedAPE_%', 'R2_brut'],
                          format_func={'RMSE_log': 'RMSE sur log (plus bas = mieux)', 'R2_log': 'R² sur log (plus haut = mieux)',
                                       'MedAPE_%': 'Erreur relative médiane % (plus bas = mieux)', 'R2_brut': 'R² échelle brute (plus haut = mieux)'}.get)
    fig = go.Figure()
    for m in MODELS:
        fig.add_bar(x=[f'{m}<br><b>{T.loc[m, metric]:.3f}</b>'], y=[T.loc[m, metric]], error_y=dict(type='data', array=[S.loc[m, metric]], color='black'),
                    marker_color=MODEL_COLORS[m], marker_pattern_shape=MODEL_PATTERNS[m], marker_line_color='black', marker_line_width=1, name=m)
    fig.update_layout(**PLOT_LAYOUT, showlegend=False, title=f'{metric} sur le jeu de test (moyenne ± écart-type sur 5 graines)', height=450)
    described_chart(fig, f"Barres de {metric} pour les deux modèles, valeur écrite sous chaque barre. Hachures = Random Forest, plein = TabM.",
                    T[['RMSE_log', 'R2_log', 'R2_brut', 'MedAPE_%', 'fit_s']].round(3), key='perf')
    st.subheader('Synthèse')
    tab = pd.DataFrame({'RMSE log – CV 5 plis': CV.CV_RMSE_log, 'RMSE log – test': T.RMSE_log, 'R² log – test': T.R2_log, 'R² brut – test': T.R2_brut})
    st.dataframe(tab.style.format('{:.3f}'), use_container_width=True)
    st.markdown(f"TabM réduit la RMSE log de la baseline de **{RES['gain_cv_pct']:.1f} %** en validation croisée et de **{RES['gain_test_pct']:.1f} %** sur le jeu de test.")
