import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
#import Explo_Viz 
from utils import Explo_Viz

# #########################
# ⚙️ LOAD & PREPROCESS ⚙️ #
##########################

@st.cache_data  
def load_and_preprocess_data():
    """Charge et prétraite les données une seule fois."""
    df_cons = pd.read_csv("../DataSet/eco2mix-regional-cons-def.csv", sep=';')
    df_cons_preprocessed = Explo_Viz.preprocess_data(df_cons)
    return df_cons_preprocessed

def main():
    st.title("Prédiction de Consommation Electrique en France")
    st.sidebar.title("Etapes du Projet")
    pages = ["Contexte du Projet", "Exploration", "DataVizualization", "Modélisation"]
    page = st.sidebar.radio("Aller vers", pages)
    st.sidebar.title("Simulateur")
    st.sidebar.page_link("pages/simulateur.py", label="📊 Prédiction Régionnale Conso Future")
    df_cons_preprocessed = load_and_preprocess_data()  # Charger les données prétraitées
    

    if page == pages[0]: 
        st.write("### Contexte")
        st.write(""" Contexte : L’adéquation entre la production et la consommation d’électricité est au cœur des préoccupations d’un acteur de l’énergie comme EDF. 
                 EDF, en tant que producteur et commercialisateur d’électricité est en effet un responsable d’équilibre vis-à-vis de RTE. 
                 Cela signifie qu’il se doit d’assurer à tout instant un équilibre entre sa production et la consommation de ses clients, sous peine de pénalités. 
                 Pour se faire, construire un modèle de prévision de la consommation de ses clients est une activité essentielle au bon fonctionnement de EDF.""") 
        st.write('Objectif : Constater le phasage entre la consommation et la production énergétique au niveau national et au niveau régional '
        'Analyse au niveau régional pour en déduire une prévision de consommation au niveau national et au niveau régional (risque de black out notamment)')
    
    elif page == pages[1]:
        st.write("### Exploration")
        st.dataframe(df_cons_preprocessed.head(10))  # Utiliser le dataframe prétraité
        st.write(df_cons_preprocessed.shape)
        st.dataframe(df_cons_preprocessed.describe())
        st.markdown("[Cliquez ici pour en savoir plus sur les actions de datacleaning](https://www.notion.so/Projet-Data-ENERGIE-PRE-PROCESSING-Fusion-Eco2mix-2-fichiers-18c725f38aa58062b1d7f79dc035f834?pvs=4)")
        

    elif page == pages[2]:
        st.header("Phasage entre consommation et production"
        )

        st.write ("""En plus de ne pas avoir le même mix energétique, les régions sont dans une situation de disparité de leurs capacités de production pour couvrir leurs besoins.
                  (cf. infra.)""")

#################################
# ⚙️ AFFICHAGES DES GRAPHS    ⚙️#
#################################

        fig2 = Explo_Viz.create_barplot(df_cons_preprocessed)
        fig2.text(0.5, -0.15, "Certaines régions sont largement déficitaires en terme de phasage entre leur production et leurs besoin. Cf. Couverture 100%", ha='center', va='top', fontsize=12)
        st.pyplot(fig2)
        plt.close(fig2)
        
        st.write("");st.write("") 
        st.write(""" Pour résoudre cela avec l'aides des opérateurs d'énergie, les régions procèdent toute l'année à des *échanges*.
                Le graphique ci-après permet de constater quelque soit la période et la maille temporelle choisie :
                la **variabilité des besoins** des Régions au fil du temps d'une part. Le phasage entre Consommation 
                 et Production au moyen des **échanges inter-régionnaux** d'autre part.
""")

## ⚙️ FILTRAGES CONSOS / REGIONS ####
        st.markdown("<hr style='border: 2px solid #4CAF50;'>", unsafe_allow_html=True)
        st.markdown('<h6 style="text-align: center; color: #4CAF50;">🔎 Filtres d\'Analyse</h6>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            annee = st.selectbox("**Année** ('None' pour tout le dataset')", options=sorted(df_cons_preprocessed['DH'].dt.year.unique(), 
            reverse=True) + [None], index=sorted(df_cons_preprocessed['DH'].dt.year.unique(), reverse=True).index(2022)  # 2022 par défaut
            )
        with col2:
            if annee:
                mois = st.selectbox("**Mois** ('None' pour toute l'année)",
                options=sorted(df_cons_preprocessed[df_cons_preprocessed['DH'].dt.year == annee]['DH'].dt.month.unique()) + [None],index=0
                )
            else:
                mois = None
        with col3:
            frequence_resample = st.radio("**Fréquence** (échantillonnage)", options=['Heure', 'Jour', 'Semaine', 'Mois'],index=1  # 'Jour' par défaut
            )

        regions_preselectionnees = ['Occitanie', 'Auvergne-Rhône-Alpes', 'Bretagne', 'Pays de la Loire']
        regions = sorted(df_cons_preprocessed['Région'].unique())

        regions_selected = st.multiselect("Régions à comparer (4 maximum)", options=regions,default=regions_preselectionnees
        )
        st.markdown("<hr style='border: 2px solid #4CAF50;'>", unsafe_allow_html=True)
#################################
        st.write("Ce graphique montre l'évolution de la consommation et de la production d'énergie par région.")
        st.write("Vous pouvez utiliser les filtres ci-dessus pour explorer différentes périodes et régions.")
        
        fig = Explo_Viz.create_regional_plots(df_cons_preprocessed, annee, mois, None, frequence_resample, regions_selected)
        st.pyplot(fig)
        plt.close(fig)

        
        st.header("Evolution de la consommation, influences exogènes")
                  

    elif page == pages[3]:
        st.write("### Modélisation")

if __name__ == "__main__":
    main()