import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import streamlit as st
import gdown
from scipy.stats import pearsonr
from scipy.stats import spearmanr

# ###############################
# ⚙️ PREPROCESSING 1 ⚙️ #
################################
def preprocess_data(df_cons):
    # (Votre code de prétraitement reste inchangé)
    df_cons['DH'] = pd.to_datetime(df_cons['Date - Heure'], errors='coerce')
    if 'Column 30' in df_cons.columns:
        df_cons.drop('Column 30', axis=1, inplace=True)
    df_cons['Eolien (MW)'] = pd.to_numeric(df_cons['Eolien (MW)'], errors='coerce')
    df_cons['Eolien (MW)'].fillna(0, inplace=True)
    df_cons.fillna(0, inplace=True)
    df_cons['Echange Import (MW)'] = df_cons['Ech. physiques (MW)'].apply(lambda x: x if x > 0 else 0)
    df_cons['Echange Export (MW)'] = df_cons['Ech. physiques (MW)'].apply(lambda x: abs(x) if x < 0 else 0)
    TCH = ['TCH Thermique (%)', 'TCH Nucléaire (%)', 'TCH Eolien (%)', 'TCH Solaire (%)', 'TCH Hydraulique (%)', 'TCH Bioénergies (%)']
    TCO = ['TCO Thermique (%)', 'TCO Nucléaire (%)', 'TCO Eolien (%)', 'TCO Solaire (%)', 'TCO Hydraulique (%)', 'TCO Bioénergies (%)']
    for col in TCH:
        df_cons[col] = df_cons.groupby('Région')[col].transform(lambda x: x.replace(0, np.nan).mean())
    for col in TCO:
        df_cons[col] = df_cons.groupby('Région')[col].transform(lambda x: x.replace(0, np.nan).mean())
    df_cons = df_cons.sort_values(by='DH')
    return df_cons

# #####################################
# ⚙️ VARIATIONS ET PHASAGES       ⚙️  #
#######################################
def create_regional_plots(df_cons_preprocessed, annee, mois, jour, frequence_resample, regions_selected):
    """Crée des graphiques comparatifs pour les régions sélectionnées (sans prétraitement)."""
# 
    prod_area = ['Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)', 'Solaire (MW)', 'Hydraulique (MW)', 'Pompage (MW)', 'Bioénergies (MW)','Echange Import (MW)','Echange Export (MW)']
    conso_line = 'Consommation (MW)'
    if mois:
        df_filtered = df_cons_preprocessed[(df_cons_preprocessed['DH'].dt.year == annee) & (df_cons_preprocessed['DH'].dt.month == mois)]
    elif annee:
        df_filtered = df_cons_preprocessed[df_cons_preprocessed['DH'].dt.year == annee]
    else:
        df_filtered = df_cons_preprocessed
    dfs_regions = []
    for region in regions_selected:
        df_region = df_filtered[df_filtered['Région'] == region].groupby('DH')[prod_area + [conso_line]].mean()
        frequency_mapping = {'Heure': 'H', 'Jour': 'D', 'Semaine': 'W', 'Mois': 'ME'}
        pandas_frequency = frequency_mapping.get(frequence_resample)
        if pandas_frequency:
            df_region = df_region.resample(pandas_frequency).mean()
        else:
            raise ValueError(f"Fréquence de rééchantillonnage invalide: {frequence_resample}")
        dfs_regions.append(df_region)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    axes = axes.flatten()

    fig.suptitle(
        "Consommations VS Moyen de Production VS Import/exports Régionnaux",
        fontsize=16, y=0.98  # Ajuste la hauteur du titre
    )

    def format_dates_by_frequency(df, ax, frequence_resample):
        if frequence_resample == 'Jour':
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        elif frequence_resample == 'Semaine':
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-W%U'))
        elif frequence_resample == 'Mois':
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        elif frequence_resample == 'Heure':
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
        else:
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.tick_params(axis='x', labelsize=6)

    xlabel = "Période selon filtre"
    ylabel = "MW (Mégawatts)"

    for i, df_region in enumerate(dfs_regions):
        colors = [
        '#1E90FF',  # Bleu foncé pour Thermique (MW) 
        '#0000FF',  # Bleu ciel pour Nucléaire (MW)
        '#4682B4',  # Bleu acier pour Eolien (MW)
        '#5F9EA0',  # Bleu cadet pour Solaire (MW)
        '#6495ED',  # Bleu clair pour Hydraulique (MW)
        '#00BFFF',  # Bleu profond pour Pompage (MW)
        '#87CEEB',  # Bleu ciel clair pour Bioénergies (MW)
        '#FF6347',  # Rouge tomate pour Echange Import (MW)
        '#32CD32',  # Vert lime pour Echange Export (MW)
    ]
        axes[i].plot(df_region.index, df_region[conso_line], label='Consommation (MW)', color='black', linewidth=3, linestyle='--')
        axes[i].stackplot(df_region.index, *[df_region[col] for col in prod_area], labels=prod_area, alpha=0.8, colors=colors)
        axes[i].set_title(f" {regions_selected[i]}", fontsize=10)
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)
        axes[i].grid(True, linestyle='--', alpha=1)
        format_dates_by_frequency(df_region, axes[i], frequence_resample)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles, labels, loc='upper center', fontsize=10, ncol=5, 
        bbox_to_anchor=(0.5, 0.95), bbox_transform=fig.transFigure
    )
    plt.subplots_adjust(top=0.85)  # Ajuste l'espace pour ne pas écraser la légende


    #plt.tight_layout()
    return fig

# ############################
# ⚙️ TAUX DE COUVERTURE    ⚙️#
##############################

def create_barplot(df_cons_preprocessed):
    """Crée un graphique en barres empilées pour le taux de couverture (sans prétraitement)."""

    prod_area = ['Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)', 'Solaire (MW)', 'Hydraulique (MW)', 'Pompage (MW)', 'Bioénergies (MW)', 'Echange Import (MW)', 'Echange Export (MW)']
    conso_line = 'Consommation (MW)'
    df_barplot = df_cons_preprocessed.groupby('Région')[prod_area + [conso_line]].sum()
    TCH = ['TCH Thermique (%)', 'TCH Nucléaire (%)', 'TCH Eolien (%)', 'TCH Solaire (%)', 'TCH Hydraulique (%)', 'TCH Bioénergies (%)']
    TCO = ['TCO Thermique (%)', 'TCO Nucléaire (%)', 'TCO Eolien (%)', 'TCO Solaire (%)', 'TCO Hydraulique (%)', 'TCO Bioénergies (%)']
    df_barplot[TCO] = df_cons_preprocessed.groupby('Région')[TCO].mean()
    df_barplot[TCH] = df_cons_preprocessed.groupby('Région')[TCH].mean()
    fig_barplot, ax = plt.subplots(figsize=(11, 7))
    df_barplot[TCO].plot.bar(stacked=True, ax=ax, alpha=0.8, rot=0)
    ax.axhline(y=100, color='black', linestyle='--', linewidth=2, label='Couverture 100%')
    ax.set_ylabel('Pourcentage (%)')
    ax.set_title('Taux de Couverture Moyen (TCO) du Mix Energétique par Région / 2013-2022')
    ax.legend(title="Sources d'énergie", loc='upper right')
    plt.tight_layout()
    ax.set_xticklabels(df_barplot.index, rotation=40, ha='center', fontsize=8, wrap=True)
    for label in ax.get_xticklabels():
        label.set_y(label.get_position()[1] - 0.07)
    return fig_barplot

# ##########################################
# ⚙️ PREPROCESSING 2 & LOAD TEMPERATURES  ⚙️#
############################################

#@st.cache_data

def compute_df_st2(df_energie):
    #Calcule df_st2 : Consommation agrégée à la maille mois et année
    return df_energie.groupby(['Année', 'Mois'])['Consommation (MW)'].sum().reset_index()

#**Aggregation à la maille plage horaire**

def aggregate_hourly_data(df_energie):
    """Agrège la consommation énergétique à la maille plage horaire."""
    
    df_st1 = df_energie[['Date', 'Saison', 'Région', 'Année', 'Mois', 'Plage Horaire', 'Consommation (MW)']]
    df_st1 = df_st1.groupby(['Date', 'Saison', 'Région', 'Année', 'Mois', 'Plage Horaire']).sum().reset_index()
    
    return df_st1  

@st.cache_data
def load_temp():
    """Télécharge et prétraite les données depuis Google Drive."""
    file_id = "1GMxi5h5sX0qaiWVayYqgdwcW-BLbZVnY"  # Ton ID de fichier extrait
    url = f"https://drive.google.com/uc?id={file_id}"  # Lien de téléchargement direct
    output = "temperature-quotidienne-regionale.csv"
    gdown.download(url, output, quiet=False)
    df_temp = pd.read_csv(output, sep=';')
    return df_temp


def preprocess_data2(df_cons):
    df_energie = df_cons.copy()
    TCH = ['TCH Thermique (%)', 'TCH Nucléaire (%)', 'TCH Eolien (%)', 'TCH Solaire (%)', 'TCH Hydraulique (%)', 'TCH Bioénergies (%)'] 
    TCO = ['TCO Thermique (%)', 'TCO Nucléaire (%)', 'TCO Eolien (%)', 'TCO Solaire (%)', 'TCO Hydraulique (%)', 'TCO Bioénergies (%)']
    colonnes_a_supprimer = ['Ech. physiques (MW)'] + TCH + TCO + [
        'Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)', 'Solaire (MW)', 'Hydraulique (MW)', 'Pompage (MW)',
        'Bioénergies (MW)', 'Stockage batterie', 'Déstockage batterie', 'Eolien terrestre', 'Eolien offshore'
    ]
    df_energie.drop(columns=colonnes_a_supprimer, inplace=True)
    df_energie.fillna(0, inplace=True)
    df_energie = df_energie.sort_values(by='DH')
    # Ajouter une nouvelle colonne avec la valeur de l'heure
    df_energie['Plage Horaire'] = df_energie['Heure'].astype(str).str.slice(0, 2).str.replace('.', '').astype(int)
    df_energie['Plage Horaire'].unique()
    # Ajouter les colonnes 'Mois' et 'Année'
    df_energie['Mois'] = pd.to_datetime(df_energie['Date']).dt.month
    df_energie['Année'] = pd.to_datetime(df_energie['Date']).dt.year
    # Fonction pour déterminer la saison en fonction du mois
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Hiver'
        elif month in [3, 4, 5]:
            return 'Printemps'
        elif month in [6, 7, 8]:
            return 'Été'
        else:
            return 'Automne'
    # Ajouter la colonne 'Saison'
    df_energie['Saison'] = df_energie['Mois'].apply(get_season)
    return df_energie

# ###############################
# ⚙️ BOX PLOT TEMPERATURES    ⚙️#
#################################


def create_boxplot(df_energie, df_temp):
    """Crée un graphique avec boxplot pour la consommation et un swarmplot pour la température moyenne par mois."""
    # Grouper par mois et année et sommer la consommation
    df_st2 = df_energie[['Année', 'Mois', 'Consommation (MW)']]
    df_st2 = df_st2.groupby(['Année', 'Mois'])['Consommation (MW)'].sum().reset_index()

    # Compiler les données du df_energie à la maille jour
    df_energie_jour = df_energie.groupby(['Région', 'Date', 'Année', 'Mois']).agg({
        'Consommation (MW)': 'sum'  
    }).reset_index()

    # Fusionner avec les données de température
    df_corr01 = pd.merge(df_energie_jour, df_temp[['Région', 'Date', 'TMoy (°C)']], on=['Région', 'Date'], how='left')

    # Suppression des années sans relevés et interpolation des valeurs manquantes
    df_corr01 = df_corr01[df_corr01['Date'] >= '2016-01-01']
    df_corr01.sort_values(by=['Région', 'Date'], inplace=True)
    df_corr01['TMoy (°C)'] = df_corr01['TMoy (°C)'].interpolate(method='linear')

    # Calculer la moyenne de TMoy par mois
    df_corr01_st2 = df_corr01.groupby(['Région', 'Année', 'Mois'])['TMoy (°C)'].mean().reset_index()
    #st.write(df_corr01_st2.sample(50))  # Remplacement de display() par st.write()
    
    # Créer la figure
    fig, ax1 = plt.subplots(figsize=(13, 8))

    # Boxplot pour la consommation par mois (à partir de df_st2, qui est l'agrégat)
    df_st2.boxplot(column='Consommation (MW)', by='Mois', grid=False, ax=ax1, 
                   positions=np.arange(len(df_st2['Mois'].unique())) - 0.2, 
                   widths=0.4, patch_artist=True, boxprops=dict(facecolor='lightyellow'))

    # Créer un second axe des ordonnées pour les températures
    ax2 = ax1.twinx()

    # Swarmplot pour la température moyenne par mois
    sns.swarmplot(x='Mois', y='TMoy (°C)', data=df_corr01_st2, ax=ax2, hue='TMoy (°C)', 
                  palette='coolwarm', size=2, legend=False)

    # Définir la palette de couleurs pour le swarmplot
    norm = plt.Normalize(df_corr01_st2['TMoy (°C)'].min(), df_corr01_st2['TMoy (°C)'].max())
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])

    # Ajouter une barre de couleur
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label('Température Moyenne (°C)')

    # Ajuster les étiquettes et le titre
    ax1.set_title('Consommation et Température Moyenne par Mois')
    ax1.set_xlabel('Mois')
    ax1.set_ylabel('Consommation (MW)')

    return fig

# ###############################
# ⚙️ BOX PLOT SAISONS        ⚙️#
#################################

def create_boxplot_season(df_st1):
    """Crée un boxplot Streamlit pour la consommation par plage horaire et saison."""
    # Grouper par Plage Horaire, Année et Saison et sommer la consommation
    df_st3 = df_st1[['Plage Horaire', 'Année', 'Saison', 'Consommation (MW)']]
    df_st3 = df_st3.groupby(['Plage Horaire', 'Année', 'Saison'])['Consommation (MW)'].sum().reset_index()

    palette = {
        'Été': '#FFD700',   
        'Hiver': '#1E90FF', 
        'Automne': '#FF8C00',
        'Printemps': '#32CD32'
    }
    fig, ax = plt.subplots(figsize=(13, 8))
    sns.boxplot(y='Consommation (MW)', x='Plage Horaire', hue='Saison', data=df_st3, showfliers=False, palette=palette, ax=ax)
    ax.set_title('Consommation par Plage Horaire en fonction de la saison')
    ax.set_xlabel('Plage Horaire')
    ax.set_ylabel('Consommation (MW)')
    return fig, df_st3

# ###############################
# ⚙️ PLOT VARIATION ANNUELLE  ⚙️#
#################################
# Créer un graphique en ligne pour la consommation par mois et année

def create_annual_plot(df_st2):
    """Crée un graphique de consommation annuelle par mois pour chaque année."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for year in df_st2['Année'].unique():
        ax.plot(df_st2[df_st2['Année'] == year]['Mois'], 
                df_st2[df_st2['Année'] == year]['Consommation (MW)'], 
                label=year)
    ax.set_title('Consommation par Mois et Année')
    ax.set_xlabel('Mois')
    ax.set_ylabel('Consommation (MW)')
    ax.legend(title='Année')
    ax.grid(True)

    return fig

# ###############################
# ⚙️     TEST STATISTIQUES    ⚙️#
#################################

def Test_corr(df_st3):
    """
    Calcule les corrélations de Spearman et Pearson entre la Plage Horaire et la Consommation électrique.
    Retourne un dictionnaire avec les coefficients et les p-values.
    """

    # Calcul des corrélations globales sur le jeu de données agrégé
    spearman_corr, spearman_p = spearmanr(df_st3['Plage Horaire'], df_st3['Consommation (MW)'])
    pearson_corr, pearson_p = pearsonr(df_st3['Plage Horaire'], df_st3['Consommation (MW)'])

    # Retourner figure + corrélations dans un dict pour affichage en aval
    corr_results = {
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
        "pearson_corr": pearson_corr,
        "pearson_p": pearson_p
    }

    return corr_results, df_st3

