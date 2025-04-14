import streamlit as st
import pandas as pd
from datetime import datetime
import joblib
import os
import seaborn as sns
import plotly.express as px

# Titre de la page
st.title("Simulateur de Consommation Future")

st.markdown("<hr style='border: 2px solid #4CAF50;'>", unsafe_allow_html=True)
st.markdown('<h5 style="text-align: center; color: #4CAF50;">🔎 Votre besoin de prévision</h5>', unsafe_allow_html=True)

# Affichage des sélecteurs de date
# Création de deux colonnes
col1, col2 = st.columns(2)
# Affichage des sélecteurs de date dans les colonnes
with col1:
    start_date = st.date_input("Date de début", datetime.today(), key="start_date")
with col2:
    end_date = st.date_input("Date de fin", datetime.today(), key="end_date")

# Affichage du sélecteur de Régions
options = ["Auvergne-Rhône-Alpes", "Bourgogne-Franche-Comté",
           "Bretagne", "Centre-Val de Loire", "Grand Est", "Hauts-de-France",
           "Normandie", "Nouvelle-Aquitaine", "Occitanie", "Pays de la Loire",
           "Provence-Alpes-Côte d'Azur", "Île-de-France"]
choix_liste = st.multiselect("Sélectionnez la(les) Région(s) :", options)

# Affichage du sélecteur de Modèle
selected_model = st.selectbox("Sélectionnez le modèle entrainé :", options= ["Random Forest", "XGBoost", "Prophet"])

# Récap
st.write(f"Vous avez choisi de simuler la consommation pour la période du {start_date} au {end_date}. Pour la (les) région(s) suivante(s) {choix_liste}")
st.markdown("<hr style='border: 2px solid #4CAF50;'>", unsafe_allow_html=True)


############################

# Chargement et prétraitement des données
df_future_temp = pd.read_csv("dataset_streamlit/Future_temp.csv", index_col=0)

# Conversion de la colonne 'time' en datetime
df_future_temp['time'] = pd.to_datetime(df_future_temp['time'])

# Renommage des colonnes
df_future_temp.rename(columns={'time': 'Date', 'region': 'Région', 'C°': 'TMoy (°C)'}, inplace=True)

# Ajout des colonnes temporelles depuis l'index
df_future_temp['Année'] = df_future_temp['Date'].dt.year
df_future_temp['month'] = df_future_temp['Date'].dt.month
df_future_temp['day_of_week'] = df_future_temp['Date'].dt.dayofweek
df_future_temp['day_of_year'] = df_future_temp['Date'].dt.dayofyear
df_future_temp['week_of_year'] = df_future_temp['Date'].dt.isocalendar().week

df_future_temp = df_future_temp.set_index('Date')

# Réorganiser les colonnes (si nécessaire)
df_future_temp = df_future_temp[['Région', 'TMoy (°C)', 'Année', 'month', 'day_of_week', 'day_of_year', 'week_of_year']]

# Filtrage des données pour la période sélectionnée
df_filtered = df_future_temp[(df_future_temp.index >= pd.to_datetime(start_date)) & (df_future_temp.index <= pd.to_datetime(end_date))]

# Vérification de la disponibilité des données
if df_filtered.empty:
    st.warning("⚠️ Aucune donnée disponible pour cette période. Essayez une autre date.")
else:
    predictions = []
    for region in choix_liste:
        df_region = df_filtered[df_filtered["Région"] == region]

        if df_region.empty:
            st.warning(f"⚠️ Aucune donnée pour la région {region} sur cette période.")
            continue

        # Sélection des variables explicatives
        features = ['TMoy (°C)', 'Année', 'month', 'day_of_week', 'day_of_year', 'week_of_year']
        X_region = df_region[features]

        # Chargement du bon modèle en fonction du choix de l'utilisateur
        model_choices = {
            "Random Forest": "RF",
            "XGBoost": "XGB",
            "Prophet": "Prophet"
        }

        model_name = f"{model_choices[selected_model]}_{region}.joblib"
        #model_path = f"../Modélisations_prédictions/{model_name}"
        model_path = os.path.join("Modélisations_prédictions", model_name)
        print(f"Trying to load model from: {model_path}")

        try:
            model = joblib.load(model_path)
            preds = model.predict(X_region)

            # Ajout des prédictions au DataFrame
            df_region["Consommation_Prévue (MW)"] = preds
            predictions.append(df_region[["Région", "Consommation_Prévue (MW)"]])

        except FileNotFoundError:
            st.error(f"❌ Modèle non trouvé pour la région {region}. Vérifiez les fichiers.")

    # Affichage des résultats

    if predictions:
        df_results = pd.concat(predictions)

        st.subheader("📊 Résultats de la prédiction (avec filtres)")

        #Affichage Graphique
        fig = px.line(df_results.reset_index(), x="Date", y="Consommation_Prévue (MW)", color="Région",
              title="Visualisation graphique interactive")
        
        # Centrer le titre et le mettre en vert
        fig.update_layout(
            title={
                'text': "📊 Visualisation graphique interactive",
                'x': 0.5,  # Centre horizontalement (0.0 à 1.0)
                'y': 0.95, # Position verticale (ajuster si nécessaire)
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'color': '#4CAF50'
                }
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        # Affichage du DataFrame entier avec st.data_editor
        st.markdown('<h6 style="text-align: center; color: #4CAF50;">📊 Tableau exportable des résulats</h6>', unsafe_allow_html=True)
        st.data_editor(
            df_results.reset_index(),  # on reset l'index pour avoir 'Date' comme colonne visible
            column_config={
                "Date": st.column_config.DateColumn("Date"),
                "Consommation_Prévue (MW)": st.column_config.NumberColumn("Prévision (MW)"),
                "Région": st.column_config.TextColumn("Région")
            },
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True
        )

