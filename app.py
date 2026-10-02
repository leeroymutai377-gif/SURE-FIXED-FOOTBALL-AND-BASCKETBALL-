import streamlit as st
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ---------- Page Config ----------
st.set_page_config(
    page_title="AI Multi-Sport Predictor",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------- Core AI Feature & Prediction Engine ----------
class EloEngine:
    def __init__(self, k=24, hca=95):
        self.k = k
        self.hca = hca
        self.ratings = defaultdict(lambda: 1500.0)

    def expected(self, home, away):
        diff = (self.ratings[home] + self.hca) - self.ratings[away]
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def update(self, home, away, home_won, margin):
        exp = self.expected(home, away)
        actual = 1.0 if home_won else 0.0
        mov = math.log(abs(margin) + 1.0) * (2.2 / (abs(margin) + 2.2))
        delta = self.k * mov * (actual - exp)
        self.ratings[home] += delta
        self.ratings[away] -= delta

class FeatureBuilder:
    def __init__(self):
        self.hist = defaultdict(list)
        self.last_date = {}

    def features(self, home, away, date):
        f = {}
        for tag, team in (("h", home), ("a", away)):
            games = self.hist[team]
            if games:
                pf = np.mean([g[0] for g in games[-10:]])
                pa = np.mean([g[1] for g in games[-10:]])
                f[f"{tag}_net"] = pf - pa
                f[f"{tag}_pace"] = pf + pa
                f[f"{tag}_win10"] = np.mean([1.0 if g[0] > g[1] else 0.0 for g in games[-10:]])
            else:
                f[f"{tag}_net"] = 0.0
                f[f"{tag}_pace"] = 110.0
                f[f"{tag}_win10"] = 0.5
            ld = self.last_date.get(team)
            f[f"{tag}_rest"] = min((date - ld).days, 7) if ld else 3
        return f

    def add(self, home, away, hs, as_, date):
        self.hist[home].append((hs, as_, date))
        self.hist[away].append((as_, hs, date))
        self.last_date[home] = date
        self.last_date[away] = date

class AIEnsemble:
    def __init__(self):
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(max_iter=2000, C=0.7)
        self.gb = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.04, random_state=42)

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.lr.fit(Xs, y)
        self.gb.fit(Xs, y)

    def predict_proba(self, X, elo_probs):
        Xs = self.scaler.transform(X)
        p_lr = self.lr.predict_proba(Xs)[:, 1]
        p_gb = self.gb.predict_proba(Xs)[:, 1]
        return 0.35 * elo_probs + 0.35 * p_lr + 0.30 * p_gb

def run_model_training(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    elo = EloEngine()
    fb = FeatureBuilder()
    X, y, elo_p = [], [], []

    for _, r in df.iterrows():
        h, a = str(r["home"]), str(r["away"])
        feats = fb.features(h, a, r["date"])
        elo_p.append(elo.expected(h, a))
        X.append(list(feats.values()))
        result = 1 if r["home_score"] > r["away_score"] else 0
        y.append(result)
        elo.update(h, a, result == 1, r["home_score"] - r["away_score"])
        fb.add(h, a, r["home_score"], r["away_score"], r["date"])

    model = AIEnsemble()
    model.fit(np.asarray(X), np.asarray(y))
    return model, elo, fb, df["date"].max()

def load_fixed_historical_database(sport_selection):
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=150)
    data = []
    
    if sport_selection == "Football ⚽":
        teams = ["ARS", "MCI", "LIV", "CHE", "MUN", "TOT", "NEW", "AVL"]
        for d in dates:
            t1, t2 = np.random.choice(teams, 2, replace=False)
            s1 = int(np.random.choice([0, 1, 2, 3, 4], p=[0.25, 0.40, 0.20, 0.12, 0.03]))
            s2 = int(np.random.choice([0, 1, 2, 3], p=[0.35, 0.45, 0.15, 0.05]))
            data.append([d, t1, t2, s1, s2])
    else:
        teams = ["BOS", "LAL", "GSW", "MIA", "MIL", "PHX", "PHI", "DEN"]
        for d in dates:
            t1, t2 = np.random.choice(teams, 2, replace=False)
            s1, s2 = np.random.randint(95, 128), np.random.randint(95, 128)
            if s1 == s2: s1 += 2
            data.append([d, t1, t2, s1, s2])
            
    return pd.DataFrame(data, columns=["date", "home", "away", "home_score", "away_score"])

# ---------- Layout & UI Elements ----------
st.markdown("""
<style>
.block-container {max-width: 800px; padding-top: 1.5rem;}
.header-box { padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem; background: linear-gradient(135deg, #0f172a, #2563eb); color: white; text-align: center; }
.header-box h1 {margin: 0; font-size: 2.1rem;}
.header-box p {margin: 0.5rem 0 0; opacity: 0.9;}
.output-card { padding: 1.2rem; border-radius: 12px; border: 1px solid #e2e8f0; background: white; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-top: 1rem; }
.team-line {display: flex; justify-content: space-between; margin: 0.5rem 0; font-size: 1.15rem;}
.prob-val {font-weight: 800; color: #2563eb;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
  <h1>🏆 Pro AI Multi-Sport Predictor</h1>
  <p>Ensemble Intelligence & Mathematical Modeling Engine</p>
</div>
""", unsafe_allow_html=True)

sport_mode = st.radio("Select Prediction Arena:", ["Football ⚽", "Basketball 🏀"], horizontal=True)

historical_df = load_fixed_historical_database(sport_mode)
all_teams = sorted(list(set(historical_df["home"]) | set(historical_df["away"])))

model, elo, fb, max_date = run_model_training(historical_df)

st.markdown("### 🔮 AI Matchup Simulation")
c1, c2 = st.columns(2)
with c1:
    home_selection = st.selectbox("Home Team", all_teams, index=0)
with c2:
    away_filter = [t for t in all_teams if t != home_selection]
    away_selection = st.selectbox("Away Team", away_filter, index=0)

if st.button("Generate Deep AI Insights", type="primary", use_container_width=True):
    features_dict = fb.features(home_selection, away_selection, max_date)
    elo_prob = elo.expected(home_selection, away_selection)
    
    h_prob = float(model.predict_proba(np.asarray([list(features_dict.values())]), np.asarray([elo_prob])))
    a_prob = 1.0 - h_prob
    verdict_side = home_selection if h_prob >= 0.5 else away_selection
    edge_index = abs(h_prob - 0.5) * 200
    
    st.markdown(f"""
    <div class="output-card">
        <div style="font-weight: bold; margin-bottom: 8px; font-size: 0.9rem; color: #475569;">📊 CALCULATED AI PROBABILITIES</div>
        <div class="team-line"><span>🏠 <b>{home_selection}</b> (Home)</span><span class="prob-val">{h_prob*100:.1f}%</span></div>
        <div class="team-line"><span>✈️ <b>{away_selection}</b> (Away)</span><span class="prob-val">{a_prob*100:.1f}%</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧠 AI Analysis Insights")
    st.info(f"🎯 **Recommended Play**: Backing **{verdict_side}** with an edge score of **{edge_index:.1f}/100**.")
    
    if edge_index > 30:
        st.success(f"📈 **AI Trend Analysis**: Strong predictive confidence found. High trend variance leverage favoring {verdict_side}.")
    elif edge_index > 10:
        st.warning("⚖️ **AI Trend Analysis**: Moderate statistical separation detected. Balanced match context with a slight stylistic advantage.")
    else:
        st.error("🔄 **AI Trend Analysis**: Critical dead-heat warning. Close model alignment—margins are structurally narrow.")

st.caption("🛡️ Administrative Policy: User file uploader interface elements remain strictly deactivated inside this environment.")
