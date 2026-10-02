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
    page_title="AI Multi-Sport Predictor Pro",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------- Secure Local Persistent Memory Storage ----------
# Streamlit clears data on refresh; we use session state as a mock database table for your VVIP posts.
if "vvip_posts" not in st.session_state:
    st.session_state.vvip_posts = [
        {"date": "2026-10-03", "sport": "Football ⚽", "matchup": "Arsenal vs Chelsea", "pick": "Arsenal Win", "odds": "1.85", "confidence": "94%"},
        {"date": "2026-10-03", "sport": "Basketball 🏀", "matchup": "LA Lakers vs Golden State", "pick": "Over 218.5 Points", "odds": "1.90", "confidence": "91%"}
    ]

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
                pf = np.mean([g for g in games[-10:]])
                pa = np.mean([g for g in games[-10:]])
                f[f"{tag}_net"] = pf - pa
                f[f"{tag}_pace"] = pf + pa
                f[f"{tag}_win10"] = np.mean([1.0 if g > g else 0.0 for g in games[-10:]])
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
        teams = ["Arsenal", "Man City", "Liverpool", "Chelsea", "Man United", "Real Madrid", "Barcelona", "Bayern Munich"]
        for d in dates:
            t1, t2 = np.random.choice(teams, 2, replace=False)
            s1 = int(np.random.choice(, p=[0.25, 0.40, 0.20, 0.12, 0.03]))
            s2 = int(np.random.choice(, p=[0.35, 0.45, 0.15, 0.05]))
            data.append([d, t1, t2, s1, s2])
    else:
        teams = ["LA Lakers", "Boston Celtics", "Golden State", "Miami Heat", "Milwaukee Bucks", "Phoenix Suns", "Phila 76ers", "Denver Nuggets"]
        for d in dates:
            t1, t2 = np.random.choice(teams, 2, replace=False)
            s1, s2 = np.random.randint(95, 128), np.random.randint(95, 128)
            if s1 == s2: s1 += 2
            data.append([d, t1, t2, s1, s2])
            
    return pd.DataFrame(data, columns=["date", "home", "away", "home_score", "away_score"])

# ---------- Luxury Theme CSS Styling ----------
st.markdown("""
<style>
.stApp { background-color: #0f172a; color: #f8fafc; }
.header-box { 
    padding: 1.5rem; border-radius: 20px; margin-bottom: 1.5rem; 
    background: linear-gradient(135deg, #1e1b4b, #4338ca); 
    color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.header-box h1 {margin: 0; font-size: 2rem; font-weight: 800;}
.header-box p {margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem;}
.vvip-card {
    padding: 1.2rem; border-radius: 14px; border-left: 5px solid #eab308;
    background: #1e293b; margin-bottom: 1rem; box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
.output-card { 
    padding: 1.2rem; border-radius: 12px; border: 1px solid #334155; 
    background: #1e293b; margin-top: 1rem; 
}
.team-line {display: flex; justify-content: space-between; margin: 0.4rem 0; font-size: 1.15rem;}
.prob-val {font-weight: 800; color: #818cf8;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
  <h1>🏆 Pro AI Multi-Sport Dashboard</h1>
  <p>Admin VVIP Management & Automated Bulk Match Analytics</p>
</div>
""", unsafe_allow_html=True)

# Main Navigation Hub
tab_vvip, tab_bulk, tab_single, tab_admin = st.tabs([
    "🔥 VVIP Sure Odds", 
    "📊 Bulk AI Analytics", 
    "🔮 Single Matchup Simulation", 
    "🛡️ Admin Portal"
])

# Global Sport Mode Data Load
sport_mode = st.radio("Select Prediction Arena:", ["Football ⚽", "Basketball 🏀"], horizontal=True)
historical_df = load_fixed_historical_database(sport_mode)
all_teams = sorted(list(set(historical_df["home"]) | set(historical_df["away"])))
model, elo, fb, max_date = run_model_training(historical_df)

# ================= TAB 1: VVIP SURE ODDS BOARD =================
with tab_vvip:
    st.markdown("### 🌟 Admin Featured VVIP Sure Odds")
    st.caption("Handpicked selections published directly by the system platform administrator.")
    
    active_posts = [p for p in st.session_state.vvip_posts if p["sport"] == sport_mode]
    
    if not active_posts:
        st.info("No VVIP tips posted for this arena today. Check back later!")
    else:
        for post in active_posts:
            st.markdown(f"""
            <div class="vvip-card">
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#eab308; font-weight:bold; margin-bottom:5px;">
                    <span>📅 {post['date']}</span> <span>SURE ODDS VVIP ★</span>
                </div>
                <div style="font-size:1.3rem; font-weight:bold; margin-bottom:5px;">{post['matchup']}</div>
                <div style="font-size:1.1rem; color:#f8fafc;">🎯 Target Prediction: <span style="color:#22c55e; font-weight:bold;">{post['pick']}</span></div>
                <div style="margin-top:5px; font-size:1rem; color:#94a3b8;">
                    Odds Value: <b>{post['odds']}</b> | AI Core Model Confidence: <span style="color:#818cf8;"><b>{post['confidence']}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ================= TAB 2: BULK AI ANALYTICS =================
with tab_bulk:
    st.markdown("### 📊 Automated Multi-Match Matrix Analytics")
    st.caption("Instantly simulate all upcoming variations across the league using optimized machine learning algorithms.")
    
    if st.button("🚀 Run Complete League Simulation", type="primary", use_container_width=True):
        bulk_results = []
        # Generate combinatorics matrix paths dynamically
        for i, h_team in enumerate(all_teams):
            for j, a_team in enumerate(all_teams):
                if h_team != a_team:
                    f_dict = fb.features(h_team, a_team, max_date)
                    e_prob = elo.expected(h_team, a_team)
                    h_prob = float(model.predict_proba(np.asarray([list(f_dict.values())]), np.asarray([e_prob])))
                    
                    verdict = h_team if h_prob >= 0.5 else a_team
                    edge_val = abs(h_prob - 0.5) * 200
                    
                    bulk_results.append({
                        "Home Team": h_team,
                        "Away Team": a_team,
                        "Home Win %": f"{h_prob*100:.1f}%",
                        "Away Win %": f"{(1-h_prob)*100:.1f}%",
                        "AI Target Pick": verdict,
                        "Edge Rating": round(edge_val, 1)
                    })
        
        bulk_df = pd.DataFrame(bulk_results).sort_values(by="Edge Rating", ascending=False)
        st.success("Matrix complete. Matchups sorted by peak statistical algorithm advantage:")
        st.dataframe(bulk_df, use_container_width=True, hide_index=True)

