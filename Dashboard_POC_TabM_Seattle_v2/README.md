# Dashboard – PoC TabM sur les émissions des bâtiments de Seattle (2016)

Preuve de concept : prédire les **émissions de gaz à effet de serre** d'un bâtiment non résidentiel de Seattle à partir de ses seules
données structurelles, avec **TabM** (Gorishniy, Kotelnikov & Babenko, ICLR 2025, [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)),
comparé au Random Forest de mon projet P3 (baseline).

## Sections
1. **Exploration des données** : statistiques descriptives, comptages, distribution des émissions, nuage surface/émissions, émissions médianes par quartier, corrélations.
2. **Prédiction** : choix d'un bâtiment du jeu de test ou saisie de ses caractéristiques → émissions prédites par TabM, comparaison avec la baseline, explication SHAP locale.
3. **Comparaison des modèles** : métriques sur le jeu de test (5 graines) et en validation croisée.

## Accessibilité (WCAG 2.1)
- Palette Okabe-Ito / cividis (daltonisme) ; la couleur n'est jamais seule : formes de marqueurs, motifs de barres, valeurs écrites (1.4.1).
- Chaque graphique a un titre, une description textuelle et ses données en tableau (1.1.1).
- Texte et graduations foncés sur fond clair, contraste > 4,5:1 (1.4.3, 1.4.11) ; thème clair imposé par `.streamlit/config.toml`.
- Clavier (2.1.1) : les trois écrans sont des liens en haut de page (Tab puis Entrée, une URL par écran) ; barres d'outils des
  graphiques et grilles de données interactives retirées (tableaux HTML simples), soit environ 16 arrêts de tabulation sur
  l'écran d'exploration au lieu de plus de 80 ; ordre de tabulation du formulaire identique à l'ordre visuel.
- Défilement : Espace, Page suivante / précédente, flèches, Début / Fin font défiler la page même sans élément focalisé ;
  lien « Aller au contenu principal » (2.4.1) ; focus visible (2.4.7) ; mode d'emploi clavier dans la barre latérale.
- Pas de carte interactive (zoom et survol à la souris, illisible au clavier) : remplacée par les émissions médianes par
  quartier en barres triées, avec valeurs écrites.
- Champs avec libellés et aides (3.3.2) ; parts d'énergie saisies en % dans des champs numériques plutôt qu'avec des curseurs.

## Lancer en local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Contenu
- `app.py` : application Streamlit ; `seattle_prep.py` : préparation des variables (partagée avec le notebook).
- `artifacts/` : TabM exporté en ONNX (inférence sans PyTorch), prétraitement, prédictions du jeu de test, résultats.

Données : [Seattle Building Energy Benchmarking](https://data.seattle.gov/Built-Environment/Building-Energy-Benchmarking-Data-2015-Present/teqw-tu6e/about_data) (2016).
