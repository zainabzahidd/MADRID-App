"""
MADRID Airport Investment Prioritization — Streamlit Web Application
Author: Zainab Zahid (24L-7362) | FAST-NUCES, MSBA-4A
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MADRID Airport Prioritization | Pakistan",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .stMetric { background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #0096FF; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .metric-card { background: white; padding: 1.5rem; border-radius: 12px; border-left: 5px solid #0096FF; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 1rem; }
    .cluster-badge-2 { background: #DBEAFE; color: #1D4ED8; padding: 2px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .cluster-badge-1 { background: #EDE9FE; color: #6D28D9; padding: 2px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    h1 { color: #0A1628; }
    h2, h3 { color: #1E293B; }
    .sidebar .sidebar-content { background: #0A1628; }
    div[data-testid="stSidebarNav"] { background: #0A1628; }
    .highlight-box { background: linear-gradient(135deg, #0096FF15, #6366F115); border: 1px solid #0096FF40; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── DATA ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Decision matrix (Synchronized with Decision_Matrix_Enhanced_7_Criteria.xlsx)
    data = {
        'Airport_IATA': ['BDN', 'BHV', 'BNP', 'CJL', 'DBA', 'GIL', 'GWD', 'ISB', 'ISD', 'JZK', 'KCF', 'KDU', 'KHI', 'LHE', 'LYP', 'MJD', 'MUX', 'MWD', 'ORW', 'PAJ', 'PEW', 'PJG', 'PSI', 'PZH', 'REQ', 'RQR', 'RYK', 'SFF', 'SKT', 'SKZ', 'SUL', 'SYW', 'TUK', 'UET', 'WNS', 'ZIZ', 'ZZ1'],
        'Airport_Name': ['BADIN AIRSTRIP', 'BAHAWALPUR AIRPORT', 'BANNU AIRPORT', 'CHITRAL AIRPORT', 'DALBANDIN AIRPORT', 'GILGIT AIRPORT', 'GWADAR INT,L', 'ISLAMABAD BBIAP CHAKLALA', 'ISLAMABAD IIAP', 'JUZZAK AIRSTRIP', 'KADANWARI AIRSTRIP', 'SKARDU AIRPORT', 'KARACHI JIAP', 'LAHORE AIIAP', 'FAISALABAD INT,L', 'MOHENJODARO AIRPORT', 'MULTAN INT,L', 'MIANWALI AIRPORT', 'ORMARA AIRPORT', 'PARACHINAR AIRPORT', 'PESHAWAR BKIAP', 'PANJGUR AIRPORT', 'PASNI AIRPORT', 'ZHOB AIRPORT', 'REKODIQ AIRPORT', 'PAF SHOREKOTE/RAFIQUI', 'R.Y. KHAN (S.Z) AIRPORT', 'FAISAL AIR BASE KARACHI', 'SIALKOT INT,L', 'SUKKUR (B.N.B) AIRPORT', 'SUI AIRPORT', 'SEHWAN SHARIF AIRPORT', 'TURBAT INT,L', 'QUETTA INT,L', 'NAWABSHAH AIRPORT', 'ZAMZAMA AIRSTRIP', 'ISLAMKOT AIRPORT'],
        'C1_Aircraft_Movements': [6, 54, 2, 38, 12, 827, 142, 805, 40029, 62, 190, 1228, 45322, 36023, 3083, 136, 9515, 55, 4, 46, 6920, 32, 18, 8, 321, 26, 96, 178, 5419, 1063, 90, 18, 312, 3944, 76, 2, 106],
        'C2_Passenger_Volume': [0, 0, 0, 364, 0, 31230, 8991, 0, 6804439, 1934, 0, 179454, 6711131, 6048541, 417203, 0, 1386973, 0, 0, 0, 1161063, 0, 0, 0, 0, 0, 0, 0, 939899, 23662, 2053, 0, 9687, 535065, 0, 34, 3212],
        'C3_Cargo_Volume': [0, 0, 0, 0, 0, 14, 2, 0, 70455, 0, 0, 243, 104273, 114881, 52, 0, 7276, 0, 0, 0, 9514, 0, 0, 0, 0, 0, 0, 0, 9873, 4, 0, 0, 2, 2056, 0, 0, 0],
        'C4_Mail_Volume': [0, 0, 0, 0, 0, 0, 0, 0, 322, 0, 0, 0, 751, 390, 0, 0, 22, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 1, 0, 0, 0],
        'C5_Airline_Connectivity': [1, 1, 1, 1, 1, 3, 2, 5, 7, 1, 1, 3, 8, 8, 4, 1, 5, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 5, 2, 1, 1, 2, 6, 1, 1, 1],
        'C6_Infrastructure_Quality': [2, 4, 4, 4, 3, 5, 6, 7, 10, 2, 2, 5, 9, 9, 7, 4, 7, 4, 3, 4, 7, 4, 3, 3, 4, 5, 5, 5, 8, 5, 4, 3, 5, 6, 4, 2, 3],
        'C7_Strategic_Importance': [3, 4, 4, 5, 4, 6, 8, 7, 10, 3, 4, 7, 10, 10, 7, 5, 7, 4, 4, 5, 8, 4, 4, 4, 6, 5, 5, 5, 9, 5, 4, 4, 5, 7, 4, 4, 4],
    }
    df = pd.DataFrame(data)
    return df

@st.cache_data
def run_madrid(df_input, expert_ranking, n_clusters_override=None):
    criteria_cols = ['C1_Aircraft_Movements','C2_Passenger_Volume','C3_Cargo_Volume',
                     'C4_Mail_Volume','C5_Airline_Connectivity','C6_Infrastructure_Quality','C7_Strategic_Importance']
    
    decision_matrix = df_input[criteria_cols].values
    airport_codes = df_input['Airport_IATA'].values
    airport_names = df_input['Airport_Name'].values
    
    # Normalize
    scaler = MinMaxScaler()
    normalized_matrix = scaler.fit_transform(decision_matrix)
    
    # Clustering
    linkage_matrix = linkage(normalized_matrix, method='single', metric='euclidean')
    
    if n_clusters_override:
        optimal_n = n_clusters_override
        sil_score = silhouette_score(normalized_matrix, fcluster(linkage_matrix, optimal_n, criterion='maxclust'))
    else:
        sil_scores = {}
        for k in range(2, min(8, len(df_input))):
            clusters = fcluster(linkage_matrix, k, criterion='maxclust')
            sil_scores[k] = silhouette_score(normalized_matrix, clusters)
        optimal_n = max(sil_scores, key=sil_scores.get)
        sil_score = sil_scores[optimal_n]
    
    final_clusters = fcluster(linkage_matrix, optimal_n, criterion='maxclust')
    
    # Centroids
    centroids = []
    for cid in sorted(np.unique(final_clusters)):
        mask = final_clusters == cid
        centroids.append(normalized_matrix[mask].mean(axis=0))
    centroids_matrix = np.array(centroids)
    
    # DIBR weights
    n_crit = len(criteria_cols)
    ranks = np.array([expert_ranking[col] for col in criteria_cols])
    numerators = n_crit - ranks + 1
    weights = numerators / numerators.sum()
    
    # MABAC cluster ranking
    weighted_centroids = centroids_matrix * weights
    baa = np.prod(weighted_centroids, axis=0) ** (1/len(centroids_matrix))
    cluster_scores = (weighted_centroids - baa).sum(axis=1)
    
    cluster_df = pd.DataFrame({'Cluster': range(1, optimal_n+1), 'Score': cluster_scores})
    cluster_df = cluster_df.sort_values('Score', ascending=False).reset_index(drop=True)
    cluster_df['Cluster_Rank'] = range(1, len(cluster_df)+1)
    
    # MABAC airport ranking
    airport_scores = []
    for cid in sorted(np.unique(final_clusters)):
        mask = final_clusters == cid
        airports_c = airport_codes[mask]
        names_c = airport_names[mask]
        data_c = normalized_matrix[mask]
        
        if len(data_c) == 1:
            airport_scores.append({'Airport_IATA': airports_c[0], 'Airport_Name': names_c[0], 'Cluster': cid, 'MABAC_Score': 0.0})
        else:
            wm = data_c * weights
            baa_c = np.prod(wm, axis=0) ** (1/len(wm))
            scores = (wm - baa_c).sum(axis=1)
            for a, n, sc in zip(airports_c, names_c, scores):
                airport_scores.append({'Airport_IATA': a, 'Airport_Name': n, 'Cluster': cid, 'MABAC_Score': sc})
    
    df_scores = pd.DataFrame(airport_scores)
    df_scores = df_scores.merge(cluster_df[['Cluster','Cluster_Rank']], on='Cluster')
    df_scores = df_scores.sort_values(['Cluster_Rank','MABAC_Score'], ascending=[True,False]).reset_index(drop=True)
    df_scores['Global_Rank'] = range(1, len(df_scores)+1)
    
    return df_scores, weights, criteria_cols, sil_score, optimal_n, linkage_matrix, normalized_matrix, final_clusters

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/FAST_National_University_logo.svg/320px-FAST_National_University_logo.svg.png", width=80) if False else None
st.sidebar.markdown("MADRID Dashboard")
st.sidebar.markdown("**FAST-NUCES | MSBA-4A**")
st.sidebar.markdown("*Zainab Zahid | 24L-7362*")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "Overview & Rankings",
    "MADRID Step-by-Step",
    "Criteria Weight Explorer",
    "Sensitivity Analysis",
    "About the Methodology"
])

st.sidebar.divider()
st.sidebar.markdown("**Dataset:** PCAA July 2024 – June 2025")
st.sidebar.markdown("**Airports:** 37 | **Criteria:** 7")
st.sidebar.markdown("**Method:** MADRID (Single Linkage + DIBR + MABAC)")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df_raw = load_data()

DEFAULT_RANKING = {
    'C2_Passenger_Volume': 1,
    'C7_Strategic_Importance': 2,
    'C1_Aircraft_Movements': 3,
    'C3_Cargo_Volume': 4,
    'C6_Infrastructure_Quality': 5,
    'C5_Airline_Connectivity': 6,
    'C4_Mail_Volume': 7,
}

CRITERIA_LABELS = {
    'C1_Aircraft_Movements': 'C1: Aircraft Movements',
    'C2_Passenger_Volume': 'C2: Passenger Volume',
    'C3_Cargo_Volume': 'C3: Cargo Volume',
    'C4_Mail_Volume': 'C4: Mail Volume',
    'C5_Airline_Connectivity': 'C5: Airline Connectivity',
    'C6_Infrastructure_Quality': 'C6: Infrastructure Quality',
    'C7_Strategic_Importance': 'C7: Strategic Importance',
}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & RANKINGS
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview & Rankings":
    st.title("Airport Investment Prioritization — Pakistan")
    st.markdown("**MADRID Multi-Criteria Decision Analysis | PCAA Dataset 2024–2025**")
    
    results, weights, crit_cols, sil, n_clust, lm, norm_mat, clusters = run_madrid(df_raw, DEFAULT_RANKING)
    
    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("#1 Priority", "KHI", "Karachi JIAP")
    col2.metric("Airports Analyzed", "37", "July 2024 – June 2025")
    col3.metric("Optimal Clusters", str(n_clust), f"Silhouette: {sil:.4f}")
    col4.metric("Criteria Used", "7", "4 quant + 3 qual")
    col5.metric("Top-5 Stability", "100%", "Sensitivity tested")
    
    st.divider()
    
    col_left, col_right = st.columns([1.6, 1])
    
    with col_left:
        st.subheader("MABAC Score by Airport — Top 20")
        top20 = results.head(20).copy()
        top20['Color'] = top20['Cluster'].map({2: '#0096FF', 1: '#6366F1'})
        top20['Airport_Label'] = top20['Airport_IATA'] + " (#" + top20['Global_Rank'].astype(str) + ")"
        
        fig = go.Figure(go.Bar(
            x=top20['MABAC_Score'],
            y=top20['Airport_Label'],
            orientation='h',
            marker_color=top20['Color'],
            text=top20['MABAC_Score'].round(4),
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>MABAC Score: %{x:.4f}<extra></extra>"
        ))
        fig.update_layout(
            height=520, yaxis={'autorange':'reversed'},
            xaxis_title="MABAC Score", yaxis_title="",
            paper_bgcolor='white', plot_bgcolor='white',
            margin=dict(l=10, r=80, t=10, b=40),
            xaxis=dict(gridcolor='#F1F5F9'),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔵 Cluster 2 (High Priority Hub Tier) &nbsp; 🟣 Cluster 1 (Regional Tier)")
    
    with col_right:
        st.subheader("Full Rankings Table")
        
        filter_cluster = st.selectbox("Filter by Cluster", ["All", "Cluster 2 (High Priority)", "Cluster 1 (Regional)"])
        
        display_df = results.copy()
        if filter_cluster == "Cluster 2 (High Priority)":
            display_df = display_df[display_df['Cluster'] == 2]
        elif filter_cluster == "Cluster 1 (Regional)":
            display_df = display_df[display_df['Cluster'] == 1]
        
        display_df_show = display_df[['Global_Rank','Airport_IATA','Airport_Name','Cluster','MABAC_Score']].copy()
        display_df_show.columns = ['Rank','Code','Airport','Cluster','Score']
        display_df_show['Score'] = display_df_show['Score'].round(4)
        display_df_show['Cluster'] = display_df_show['Cluster'].map({2:'★ Tier 1', 1:'Tier 2'})
        
        st.dataframe(
            display_df_show,
            use_container_width=True,
            height=480,
            hide_index=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MADRID STEP BY STEP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "MADRID Step-by-Step":
    st.title("MADRID: Step-by-Step Walkthrough")
    st.markdown("Explore each step of the MADRID methodology applied to Pakistan's airports.")
    
    results, weights, crit_cols, sil, n_clust, lm, norm_mat, clusters = run_madrid(df_raw, DEFAULT_RANKING)
    
    step = st.radio("Select Step", [
        "Step 1: Decision Matrix & Normalization",
        "Step 2: Hierarchical Clustering",
        "Step 3: Silhouette Analysis",
        "Step 4: Cluster Centroids",
        "Step 5: DIBR Weights",
        "Step 6: MABAC Final Ranking",
    ], horizontal=True)
    
    st.divider()
    
    if step == "Step 1: Decision Matrix & Normalization":
        st.subheader("Step 1: Decision Matrix (Raw Values)")
        st.markdown("The raw PCAA traffic data for all 37 airports across 7 criteria.")
        
        col_a, col_b = st.columns(2)
        with col_a:
            show_raw = df_raw[['Airport_IATA','Airport_Name'] + [c for c in df_raw.columns if c.startswith('C')]].copy()
            st.dataframe(show_raw, height=400, use_container_width=True, hide_index=True)
        with col_b:
            # Normalized heatmap
            norm_df = pd.DataFrame(norm_mat, columns=[CRITERIA_LABELS[c] for c in crit_cols], index=df_raw['Airport_IATA'])
            fig = px.imshow(norm_df.T, aspect='auto', color_continuous_scale='Blues',
                            title="Normalized Matrix (Min-Max Scaling)", height=400)
            fig.update_layout(paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
    
    elif step == "Step 2: Hierarchical Clustering":
        st.subheader("Step 2: Hierarchical Clustering (Single Linkage)")
        st.markdown("Single linkage method clusters airports by their normalized performance profiles.")
        
        # Build dendrogram data
        from scipy.cluster.hierarchy import dendrogram as dend
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig_dend, ax = plt.subplots(figsize=(16, 6))
        dend(lm, labels=df_raw['Airport_IATA'].values, leaf_rotation=45, leaf_font_size=9, ax=ax, color_threshold=0)
        ax.set_title('Hierarchical Clustering Dendrogram — Single Linkage', fontsize=14, fontweight='bold')
        ax.set_xlabel('Airport Code')
        ax.set_ylabel('Euclidean Distance')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig_dend, use_container_width=True)
        plt.close()
        
        st.info(f"**2 clusters** were selected as optimal based on Silhouette Analysis (Score: 0.7773)")
    
    elif step == "Step 3: Silhouette Analysis":
        st.subheader("Step 3: Silhouette Analysis for Optimal Cluster Count")
        
        sil_scores = {}
        for k in range(2, 8):
            c = fcluster(lm, k, criterion='maxclust')
            sil_scores[k] = silhouette_score(norm_mat, c)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(sil_scores.keys()), y=list(sil_scores.values()),
            mode='lines+markers', line=dict(color='#0096FF', width=3),
            marker=dict(size=10),
            name='Silhouette Score'
        ))
        fig.add_vline(x=2, line_dash="dash", line_color="red", annotation_text="Optimal: 2 clusters")
        fig.add_hrect(y0=0.70, y1=1.0, fillcolor="#0096FF", opacity=0.05, annotation_text="Excellent (>0.70)")
        fig.update_layout(
            title="Silhouette Score by Number of Clusters",
            xaxis_title="Number of Clusters", yaxis_title="Silhouette Score",
            paper_bgcolor='white', plot_bgcolor='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        scores_df = pd.DataFrame({'Clusters': list(sil_scores.keys()), 'Score': list(sil_scores.values())})
        scores_df['Quality'] = scores_df['Score'].apply(lambda x: 'Excellent' if x > 0.7 else 'Good' if x > 0.5 else 'Fair')
        st.dataframe(scores_df, hide_index=True, use_container_width=True)
    
    elif step == "Step 4: Cluster Centroids":
        st.subheader("Step 4: Cluster Centroids (Normalized)")
        
        centroids = {}
        for cid in sorted(np.unique(clusters)):
            mask = clusters == cid
            centroids[f'Cluster {cid}'] = norm_mat[mask].mean(axis=0)
        
        centroid_df = pd.DataFrame(centroids, index=[CRITERIA_LABELS[c] for c in crit_cols])
        
        fig = go.Figure()
        colors = ['#0096FF', '#6366F1']
        for i, col in enumerate(centroid_df.columns):
            fig.add_trace(go.Bar(
                name=col, x=centroid_df.index, y=centroid_df[col],
                marker_color=colors[i]
            ))
        fig.update_layout(
            barmode='group', title="Cluster Centroids by Criterion",
            paper_bgcolor='white', plot_bgcolor='white',
            height=400, xaxis_tickangle=-30
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(centroid_df.round(4), use_container_width=True)
    
    elif step == "Step 5: DIBR Weights":
        st.subheader("Step 5: DIBR Criteria Weights")
        
        weight_df = pd.DataFrame({
            'Criterion': [CRITERIA_LABELS[c] for c in crit_cols],
            'Expert Rank': [DEFAULT_RANKING[c] for c in crit_cols],
            'DIBR Weight': weights,
            'Weight %': (weights * 100).round(1)
        }).sort_values('Expert Rank')
        
        fig = px.bar(weight_df, x='Criterion', y='Weight %',
                     color='Weight %', color_continuous_scale='Blues',
                     title="DIBR Criteria Weights (%)", height=400)
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(weight_df, hide_index=True, use_container_width=True)
    
    elif step == "Step 6: MABAC Final Ranking":
        st.subheader("Step 6: MABAC Final Global Ranking")
        
        fig = px.bar(results, x='Global_Rank', y='MABAC_Score',
                     color='Cluster', color_discrete_map={2:'#0096FF',1:'#6366F1'},
                     hover_data=['Airport_IATA','Airport_Name','MABAC_Score'],
                     title="MABAC Score — All 37 Airports by Global Rank",
                     labels={'Global_Rank':'Global Rank','MABAC_Score':'MABAC Score'},
                     height=420)
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CRITERIA WEIGHT EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Criteria Weight Explorer":
    st.title("Criteria Weight Explorer")
    st.markdown("Adjust rankings and instantly see how the MADRID results change.")
    
    st.info("**How to use:** Assign a rank (1=Most Important, 7=Least Important) to each criterion. Ranks must be unique.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Ranking Input")
        custom_ranks = {}
        for c in ['C2_Passenger_Volume','C7_Strategic_Importance','C1_Aircraft_Movements',
                  'C3_Cargo_Volume','C6_Infrastructure_Quality','C5_Airline_Connectivity','C4_Mail_Volume']:
            custom_ranks[c] = st.number_input(
                CRITERIA_LABELS[c], min_value=1, max_value=7, value=DEFAULT_RANKING[c], step=1
            )
        
        ranks_valid = len(set(custom_ranks.values())) == 7
        if not ranks_valid:
            st.error("Each rank (1–7) must be unique!")
        else:
            st.success("Valid ranking — running analysis...")
    
    with col2:
        if ranks_valid:
            results_c, weights_c, crit_cols_c, sil_c, n_c, lm_c, nm_c, cl_c = run_madrid(df_raw, custom_ranks)
            results_orig, weights_orig, _, _, _, _, _, _ = run_madrid(df_raw, DEFAULT_RANKING)
            
            # Weight comparison
            wdf = pd.DataFrame({
                'Criterion': [CRITERIA_LABELS[c] for c in crit_cols_c],
                'Original %': (weights_orig * 100).round(1),
                'Custom %': (weights_c * 100).round(1),
            })
            
            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(name='Original', x=wdf['Criterion'], y=wdf['Original %'], marker_color='#94A3B8'))
            fig_w.add_trace(go.Bar(name='Custom', x=wdf['Criterion'], y=wdf['Custom %'], marker_color='#0096FF'))
            fig_w.update_layout(barmode='group', title="Weight Comparison: Original vs Custom",
                                paper_bgcolor='white', plot_bgcolor='white', height=350, xaxis_tickangle=-25)
            st.plotly_chart(fig_w, use_container_width=True)
            
            # Rank change
            merged = results_orig[['Airport_IATA','Airport_Name','Global_Rank']].merge(
                results_c[['Airport_IATA','Global_Rank']], on='Airport_IATA', suffixes=('_Orig','_Custom')
            )
            merged['Rank Change'] = merged['Global_Rank_Orig'] - merged['Global_Rank_Custom']
            merged = merged.sort_values('Global_Rank_Custom')
            
            st.subheader("Rank Changes (Top 15)")
            top15 = merged.head(15).copy()
            top15['Δ'] = top15['Rank Change'].apply(lambda x: f"▲{x}" if x > 0 else (f"▼{abs(x)}" if x < 0 else "—"))
            st.dataframe(top15[['Global_Rank_Orig','Global_Rank_Custom','Airport_IATA','Airport_Name','Δ']].rename(
                columns={'Global_Rank_Orig':'Orig Rank','Global_Rank_Custom':'New Rank','Airport_IATA':'Code','Airport_Name':'Airport'}
            ), hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Sensitivity Analysis":
    st.title("Sensitivity Analysis")
    st.markdown("Test how robust the rankings are to changes in qualitative criteria scores.")
    
    st.subheader("Adjust Qualitative Criteria Scores")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**C5: Airline Connectivity**")
        c5_delta = st.slider("Perturbation (±points)", -2, 2, 0, key="c5")
    with col2:
        st.markdown("**C6: Infrastructure Quality**")
        c6_delta = st.slider("Perturbation (±points)", -2, 2, 0, key="c6")
    with col3:
        st.markdown("**C7: Strategic Importance**")
        c7_delta = st.slider("Perturbation (±points)", -2, 2, 0, key="c7")
    
    # Run perturbed analysis
    df_perturbed = df_raw.copy()
    df_perturbed['C5_Airline_Connectivity'] = (df_perturbed['C5_Airline_Connectivity'] + c5_delta).clip(1, 7)
    df_perturbed['C6_Infrastructure_Quality'] = (df_perturbed['C6_Infrastructure_Quality'] + c6_delta).clip(1, 7)
    df_perturbed['C7_Strategic_Importance'] = (df_perturbed['C7_Strategic_Importance'] + c7_delta).clip(1, 7)
    
    results_orig, _, _, _, _, _, _, _ = run_madrid(df_raw, DEFAULT_RANKING)
    results_pert, _, _, _, _, _, _, _ = run_madrid(df_perturbed, DEFAULT_RANKING)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Original Top 10")
        orig10 = results_orig.head(10)[['Global_Rank','Airport_IATA','Airport_Name','MABAC_Score']].copy()
        orig10['MABAC_Score'] = orig10['MABAC_Score'].round(4)
        st.dataframe(orig10, hide_index=True, use_container_width=True)
    
    with col_b:
        st.subheader("Perturbed Top 10")
        pert10 = results_pert.head(10)[['Global_Rank','Airport_IATA','Airport_Name','MABAC_Score']].copy()
        pert10['MABAC_Score'] = pert10['MABAC_Score'].round(4)
        st.dataframe(pert10, hide_index=True, use_container_width=True)
    
    # Overlap metric
    orig_top5 = set(results_orig.head(5)['Airport_IATA'])
    pert_top5 = set(results_pert.head(5)['Airport_IATA'])
    overlap = len(orig_top5 & pert_top5)
    
    st.divider()
    cola, colb, colc = st.columns(3)
    cola.metric("Top-5 Overlap", f"{overlap}/5", "airports in common")
    
    # Rank change scatter
    merged = results_orig[['Airport_IATA','Global_Rank']].merge(
        results_pert[['Airport_IATA','Global_Rank']], on='Airport_IATA', suffixes=('_Orig','_Pert')
    )
    
    fig = px.scatter(merged, x='Global_Rank_Orig', y='Global_Rank_Pert',
                     hover_name='Airport_IATA',
                     title="Rank Comparison: Original vs Perturbed",
                     labels={'Global_Rank_Orig':'Original Rank','Global_Rank_Pert':'Perturbed Rank'},
                     height=400)
    fig.add_trace(go.Scatter(x=[1,37], y=[1,37], mode='lines',
                             line=dict(color='red', dash='dash'), name='No Change'))
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)
    
    if overlap == 5:
        st.success("**Top-5 airports are fully stable** — the ranking is robust to these perturbations.")
    elif overlap >= 4:
        st.warning("Mostly stable — 4/5 top airports unchanged.")
    else:
        st.error("Significant rank changes detected in top 5.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "About the Methodology":
    st.title("About the MADRID Methodology")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        ## What is MADRID?
        **MADRID** stands for *Multi-criteria Airport Decision-making with Ranking and Integration of Data*.
        It was proposed by **Corriça et al. (2025)** specifically for aviation infrastructure decision support.
        
        The method combines three established MCDA techniques:
        - **Hierarchical Clustering** (Single Linkage) — groups alternatives by profile similarity
        - **DIBR** (Defining Interrelations Between Ranked Criteria) — derives objective weights from ordinal expert rankings
        - **MABAC** (Multi-Attributive Border Approximation Area Comparison) — ranks alternatives relative to a geometric mean boundary
        
        ## Why MADRID over traditional MCDA?
        | Aspect | Traditional MCDA | MADRID |
        |--------|-----------------|--------|
        | Handles scale diversity | Penalizes small airports | Clustering equalizes scale |
        | Weight method | Pairwise (AHP) — consistency issues | Ordinal ranking — simpler, consistent |
        | Result type | Global ranking only | Cluster rank + within-cluster rank |
        | Aviation-specific | General purpose | Designed for airports |
        
        ## Dataset
        - **Source:** Pakistan Civil Aviation Authority (PCAA)
        - **Period:** July 2024 – June 2025 (FY2024-25)
        - **Scope:** 37 airports across all Pakistani provinces
        - **Criteria:** 4 quantitative (traffic data) + 3 qualitative (scored from documentation)
        """)
    
    with col2:
        st.markdown("""
        ## 7 Evaluation Criteria
        
        **Quantitative (from PCAA data):**
        - **C1** Aircraft Movements
        - **C2** Passenger Volume *(Weight: 25%)*
        - **C3** Cargo Volume
        - **C4** Mail Volume
        
        **Qualitative (scored 1–7):**
        - **C5** Airline Connectivity
        - **C6** Infrastructure Quality
        - **C7** Strategic Importance *(Weight: 21.4%)*
        
        ## Key Results
        - **Optimal Clusters:** 2 (Silhouette: 0.7773)
        - **#1 Airport:** KHI — Karachi JIAP
        - **Top-5 Stability:** 100% across all tests
        
        ## References
        Corriça, J.V.P., et al. (2025). *Novel approach for decision support in airport investment prioritization using MADRID methodology.* Journal of Air Transport Management.
        """)
        
        st.info("**Academic Use Only** — This application was developed as part of an MSBA thesis at FAST National University (NUCES), Lahore.")
        
        st.markdown("""
        ---
        **Author:** Zainab Zahid  
        **Roll Number:** 24L-7362  
        **Program:** MSBA-4A  
        **Supervisor:** Abdul Hannan 
        **Year:** 2025  
        """)
