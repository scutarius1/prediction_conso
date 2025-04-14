import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# ###############################
# ⚙️ FONCTION  PREPROCESSING ⚙️ #
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
# ⚙️ FONCTION VARIATIONS ET PHASAGES ⚙️#
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

# #####################################
# ⚙️ FONCTION TAUX DE COUVERTURE    ⚙️#
#######################################

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