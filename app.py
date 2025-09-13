import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie
import seaborn as sns
import matplotlib.pyplot as plt
import json

# -------------------------------------------------------
# Helper: Load Lottie Animations (with error handling)
# -------------------------------------------------------
def load_lottie_local(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading Lottie file: {e}")
        return None

# Load animations from local files
music_anim = load_lottie_local(r"D:\Guvi projects\AmazonMusicClustering\lotties\music-group.json")
beat_loader = load_lottie_local(r"D:\Guvi projects\AmazonMusicClustering\lotties\player music.json")
particles_anim = load_lottie_local(r"D:\Guvi projects\AmazonMusicClustering\lotties\pulsing neon heart.json")

# -------------------------------------------------------
# Load Data
# -------------------------------------------------------
try:
    df = pd.read_csv(r"D:\Guvi projects\AmazonMusicClustering\Datasets\clusterd_amazon_music.csv")
    req_df = pd.read_csv(r"D:\Guvi projects\AmazonMusicClustering\Datasets\features.csv")
except FileNotFoundError as e:
    st.error(f"⚠️ Dataset not found: {e}")
    st.stop()

# -------------------------------------------------------
# Recommendation Function
# -------------------------------------------------------
def recommend(song_name, top_n=5):
    if song_name not in df['name_song'].values:
        return pd.DataFrame()

    idx = df[df['name_song'] == song_name].index[0]
    cluster_id = df.loc[idx, 'Clusters']
    cluster_mates = df[df['Clusters'] == cluster_id]

    cosin_sim = cosine_similarity(
        req_df.iloc[idx].values.reshape(1, -1),
        req_df.loc[cluster_mates.index]
    ).flatten()

    similar_indices = cosin_sim.argsort()[-top_n-1:-1][::-1]
    recommendations = cluster_mates.iloc[similar_indices]

    return recommendations[['name_song','genres','name_artists','Cluster_Name','Clusters']]

# -------------------------------------------------------
# Inject Custom CSS + Animations
# -------------------------------------------------------
def local_css():
    st.markdown("""
    <style>
    /* Global Background with animated gradient */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphism containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }

    /* Neon glowing title */
    h1 {
        color: #fff;
        text-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px #e60073, 0 0 40px #e60073;
        font-weight: 800;
        letter-spacing: 2px;
        text-align: center;
        animation: glow 2s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { text-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px #e60073, 0 0 40px #e60073; }
        to { text-shadow: 0 0 20px #fff, 0 0 30px #ff4da6, 0 0 40px #ff4da6, 0 0 50px #ff4da6; }
    }

    /* Button styling */
    div.stButton > button {
        background: linear-gradient(45deg, #f093fb, #f5576c);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 50px;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(245, 87, 108, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.6);
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(245, 87, 108, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(245, 87, 108, 0); }
        100% { box-shadow: 0 0 0 0 rgba(245, 87, 108, 0); }
    }

    /* Selectbox styling */
    div.stSelectbox > div > div {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
    }

    /* Dataframe styling */
    .dataframe {
        background: rgba(0,0,0,0.2) !important;
        border-radius: 10px;
        color: white !important;
        backdrop-filter: blur(5px);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Markdown song cards */
    .song-card {
        padding: 15px;
        margin: 10px 0;
        background: rgba(255,255,255,0.1);
        border-left: 5px solid #f5576c;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .song-card:hover {
        background: rgba(255,255,255,0.2);
        transform: translateX(10px);
    }

    /* Footer */
    footer {
        visibility: hidden;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        color: rgba(255,255,255,0.7);
        padding: 10px 0;
        font-size: 14px;
    }

    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------
# Page Config + Inject CSS
# -------------------------------------------------------
st.set_page_config(page_title="🎶 Neon Beats AI Explorer", layout="wide", page_icon="🎧")
local_css()  # Inject custom styles

# -------------------------------------------------------
# Hero Section with Lottie
# -------------------------------------------------------
col1, col2 = st.columns([1, 2])
with col1:
    if music_anim:
        st_lottie(music_anim, height=300, key="hero_music")
    else:
        st.image("https://via.placeholder.com/300x300/ff6b6b/ffffff?text=🎵", use_column_width=True)
with col2:
    st.title("🎧 Amazon Beats AI Explorer")
    st.markdown("""
    <h4 style='text-align: center; color: white; text-shadow: 0 0 10px rgba(255,255,255,0.7);'>
    Discover music clusters & get AI-powered recommendations 🚀
    </h4>
    """, unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------------
# Tabs
# -------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Cluster Visualization", "🎵 Recommendation Engine"])

# ------------------- Tab 1: Cluster Visualization -------------------
with tab1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.header("📊 Clustered Universe of Music")

    try:
        finaldf = pd.read_csv(r"D:\Guvi projects\AmazonMusicClustering\Datasets\pca_clusters.csv")

        fig, ax = plt.subplots(figsize=(10,7))
        sns.scatterplot(data=finaldf, x="PC1", y="PC2", hue="Clusters", palette="husl", s=60, ax=ax)
        ax.set_facecolor('none')
        ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.set_title("🌌 PCA Cluster Galaxy", color='white', fontsize=16, fontweight='bold')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        st.pyplot(fig, transparent=True)

        st.subheader("🎵 Cluster Distribution")
        cluster_counts = df.groupby("Clusters")['name_song'].count()
        st.bar_chart(cluster_counts)

    except Exception as e:
        st.error(f"⚠️ Could not load PCA data: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------- Tab 2: Recommendation Engine -------------------
with tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.header("🎶 AI-Powered Song Recommender")

    if df.empty:
        st.error("No songs loaded. Check dataset path.")
    else:
        song_name = st.selectbox("🔥 Pick a song to explore similar vibes:", df['name_song'].unique(), key="song_select")

        if st.button("🚀 Get Recommendations", key="rec_btn"):
            # Create placeholders for dynamic content
            loader_placeholder = st.empty()
            results_placeholder = st.empty()

            # === STEP 1: SHOW LOADER ===
            with loader_placeholder.container():
                st.markdown("### 🌌 Scanning the music multiverse...")
                if beat_loader:
                    st_lottie(beat_loader, height=120, key="loading_beat")
                else:
                    # Fallback CSS loader
                    st.markdown('''
                    <div style="text-align: center; margin: 20px 0;">
                        <div class="equalizer">
                            <div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div>
                        </div>
                        <p style="color: white; margin-top: 10px;">Analyzing sonic vibrations... 🎵</p>
                    </div>
                    ''', unsafe_allow_html=True)

            # Optional: tiny pause for dramatic effect (remove if you want instant)
            import time
            time.sleep(0.8)

            # === STEP 2: RUN RECOMMENDATION ===
            results = recommend(song_name, top_n=5)

            # === STEP 3: CLEAR LOADER ===
            loader_placeholder.empty()

            # === STEP 4: SHOW RESULTS ===
            if results.empty:
                results_placeholder.warning("😔 No similar tracks found in this dimension.")
            else:
                results_placeholder.success(f"🎉 Tracks vibin with **{song_name}**")

                # Style the dataframe
                styled_df = results.style.set_properties(**{
                    'background-color': 'rgba(0,0,0,0.2)',
                    'color': 'white',
                    'border': '1px solid rgba(255,255,255,0.1)',
                    'border-radius': '8px'
                })
                st.dataframe(styled_df, use_container_width=True)

                st.markdown("### 🎧 Recommended Tracks")
                for i, row in results.iterrows():
                    st.markdown(f"""
                    <div class='song-card'>
                        <h4>🎵 {row['name_song']}</h4>
                        <p><b>Genre:</b> {row['genres']} | <b>Artist:</b> {row['name_artists']} | <b>Cluster:</b> {row['Cluster_Name']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(min((i + 1) * 20, 100), text=f"Loading track {i+1}/5")

    st.markdown("</div>", unsafe_allow_html=True)
# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    Made with ❤️ & 🎵 by your Sai sudharsan S G.
</div>
""", unsafe_allow_html=True)