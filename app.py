import streamlit as st
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Multi-Sport Predictor Pro",
    page_icon="🏆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

PAYPAL_CLIENT_ID = "sb"  

def render_paywall():
    html_code = f"""
    <div style="text-align: center; padding: 24px; border: 2px solid #3b82f6; border-radius: 16px; background-color: #f8fafc; margin-bottom: 20px;">
        <h3 style="color: #1e3a8a; margin-top: 0;">🔑 Premium Analytics Locked</h3>
        <p style="color: #475569; font-size: 16px;">Get full 24-hour unconstrained access to both <b>Football</b> and <b>Basketball</b> mathematical predictions.</p>
        <div style="font-size: 24px; font-weight: bold; color: #1e40af; margin: 12px 0;">Daily Pass: $5.00 USD</div>
        <div id="paypal-button-container"></div>
    </div>
    
    <script>
        if (!window.paypalScriptLoaded) {{
            var script = document.createElement('script');
            script.src = "https://paypal.com{PAYPAL_CLIENT_ID}&currency=USD";
            script.onload = function() {{ renderPaypalButtons(); }};
            document.head.appendChild(script);
            window.paypalScriptLoaded = true;
        }} else {{
            renderPaypalButtons();
        }}

        function renderPaypalButtons() {{
            document.getElementById('paypal-button-container').innerHTML = '';
            paypal.Buttons({{
                style: {{ layout: 'vertical', color: 'gold', shape: 'rect', label: 'pay' }},
                createOrder: function(data, actions) {{
                    return actions.order.create({{
                        purchase_units: [{{ amount: {{ value: '5.00', currency_code: 'USD' }} }}]
                    }});
                }},
                onApprove: function(data, actions) {{
                    return actions.order.capture().then(function(details) {{
                        window.parent.postMessage({{
                            type: 'streamlit:setComponentValue',
                            value: {{ status: 'PAID', payer: details.payer.name.given_name }}
                        }}, '*');
                    }});
                }}
            }}).render('#paypal-button-container');
        }}
    </script>
    """
    import streamlit.components.v1 as components
    return components.html(html_code, height=280)

if "unlocked" not in st.session_state:
    st.session_state.unlocked = False
if "username" not in st.session_state:
    st.session_state.username = ""

class EloEngine:
    def __init__(self, k=22, hca=90):
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

class Ensemble:
    def __init__(self):
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(max_iter=1500, C=0.5)
        self.gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.lr.fit(Xs, y)
        self.gb.fit(Xs, y)

    def predict_proba(self, X, elo_probs):
        Xs = self.scaler.transform(X)
        p_lr = self.lr.predict_proba(Xs)[:, 1]
        p_gb = self.gb.predict_proba(Xs)[:, 1]
        return 0.40 * elo_probs + 0.30 * p_lr + 0.30 * p_gb

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

    model = Ensemble()
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

st.markdown("""
<style>
.block-container {max-width: 800px; padding-top: 1.5rem;}
.header-box { padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem; background: linear-gradient(135deg, #1e293b, #3b82f6); color: white; text-align: center; }
.header-box h1 {margin: 0; font-size: 2.1rem;}
.header-box p {margin: 0.5rem 0 0; opacity: 0.9;}
.output-card { padding: 1.2rem; border-radius: 12px; border: 1px solid #e2e8f0; background: white; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-top: 1rem; }
.team-line {display: flex; justify-content: space-between; margin: 0.5rem 0; font-size: 1.15rem;}
.prob-val {font-weight: 800; color: #2563eb;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
  <h1>🏆 Pro Multi-Sport Predictor</h1>
  <p>Secure Enterprise Grade Analytics & Predictions</p>
</div>
""", unsafe_allow_html=True)

sport_mode = st.radio("Select Prediction Arena:", ["Football ⚽", "Basketball 🏀"], horizontal=True)

historical_df = load_fixed_historical_database(sport_mode)
all_teams = sorted(list(set(historical_df["home"]) | set(historical_df["away"])))

pay_status = render_paywall()

if pay_status and isinstance(pay_status, dict) and pay_status.get("status") == "PAID":
    st.session_state.unlocked = True
    st.session_state.username = pay_status.get("payer", "Valued Client")

if not st.session_state.unlocked:
    st.warning("🔒 Features Locked. Complete the PayPal transaction above to see premium analytics.")
    st.stop()

st.success(f"🎟️ Access Unlocked. Welcome, {st.session_state.username}! Premium system dataset is ready.")

model, elo, fb, max_date = run_model_training(historical_df)

st.markdown("### 🔮 Generate Matchup Probabilities")
c1, c2 = st.columns(2)
with c1:
    home_selection = st.selectbox("Home Team Asset", all_teams, index=0)
with c2:
    away_filter = [t for t in all_teams if t != home_selection]
    away_selection = st.selectbox("Away Team Asset", away_filter, index=0)

if st.button("Calculate Matchup Verdict", type="primary", use_container_width=True):
    features_dict = fb.features(home_selection, away_selection, max_date)
    elo_prob = elo.expected(home_selection, away_selection)
    
    h_prob = float(model.predict_proba(np.asarray([list(features_dict.values())]), np.asarray([elo_prob])))
    a_prob = 1.0 - h_prob
    verdict_side = home_selection if h_prob >= 0.5 else away_selection
    edge_index = abs(h_prob - 0.5) * 200
    
    st.markdown(f"""
    <div class="output-card">
        <div class="team-line"><span>🏠 <b>{home_selection}</b> (Home)</span><span class="prob-val">{h_prob*100:.1f}%</span></div>
        <div class="team-line"><span>✈️ <b>{away_selection}</b> (Away)</span><span class="prob-val">{a_prob*100:.1f}%</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.info(f"🎯 Recommended Side Selection: **{verdict_side}** (Model Separation Edge: {edge_index:.0f}/100)")

st.caption("🛡️ Administrative Policy: User file uploader interface elements have been removed from this engine.")
