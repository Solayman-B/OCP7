# Dashboard – PoC TabM sur les bâtiments non résidentiels de Seattle (2016)

Preuve de concept : prédire les **émissions de GES** et la **consommation d'énergie** d'un bâtiment non résidentiel de Seattle
à partir de ses seules données structurelles. Le modèle utilisé est **TabM** (Gorishniy, Kotelnikov & Babenko, ICLR 2025,
[arXiv:2410.24210](https://arxiv.org/abs/2410.24210)), comparé au Random Forest du projet P4 et à LightGBM.

## Sections
1. **Exploration des données** : statistiques descriptives, comptages, distributions, nuage surface/émissions, carte, corrélations.
2. **Prédiction** : on choisit un bâtiment du jeu de test ou on saisit ses caractéristiques. La page affiche la prédiction TabM des GES et de l'énergie, la compare aux autres modèles et donne l'explication SHAP locale.
3. **Comparaison des modèles** : métriques sur le jeu de test (5 seeds) et intérêt de l'ENERGY STAR Score.

## Accessibilité (WCAG 2.1)
- Palette Okabe-Ito / cividis, sûre pour les daltoniens. La couleur n'est jamais seule : formes de marqueurs, motifs de barres et valeurs écrites la doublent (1.4.1).
- Chaque graphique a un titre, une description textuelle et ses données en tableau (1.1.1).
- Texte et graduations en gris foncé sur fond blanc, contraste supérieur à 4,5:1 (1.4.3, 1.4.11).
- Composants Streamlit natifs utilisables au clavier, avec un focus visible (2.1.1, 2.4.7).
- Champs de saisie avec libellés et aides (3.3.2).

## Lancer en local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement (Streamlit Community Cloud)
1. Pousser ce dossier dans un dépôt GitHub public.
2. Sur https://share.streamlit.io, cliquer sur « Create app », choisir le dépôt, la branche `main` et le fichier `app.py`.
3. Dans les « Advanced settings », choisir **Python 3.13**, puis cliquer sur « Deploy ».

## Contenu
- `app.py` : l'application Streamlit.
- `seattle_prep.py` : la préparation des variables, partagée avec le notebook.
- `artifacts/` : les modèles TabM exportés en ONNX (inférence légère sans PyTorch), le prétraitement, les prédictions et SHAP pré-calculés, ainsi que les résultats.

Données : [Seattle Building Energy Benchmarking](https://data.seattle.gov/Built-Environment/Building-Energy-Benchmarking-Data-2015-Present/teqw-tu6e/about_data) (2016).
