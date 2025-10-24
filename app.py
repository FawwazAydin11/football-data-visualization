import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Statistik Sepak Bola Dunia",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_and_clean_data():
    results = pd.read_csv("results.csv")
    goals = pd.read_csv("goalscorers.csv")
    shootouts = pd.read_csv("shootouts.csv")
    former = pd.read_csv("former_names.csv")
    
    results_clean = results.copy()
    results_clean["date"] = pd.to_datetime(results_clean["date"], errors='coerce')
    results_clean = results_clean.dropna(subset=["date"])
    results_clean["year"] = results_clean["date"].dt.year
    results_clean["home_team"] = results_clean["home_team"].str.strip().str.title()
    results_clean["away_team"] = results_clean["away_team"].str.strip().str.title()
    results_clean[["home_score", "away_score"]] = results_clean[["home_score", "away_score"]].fillna(0)
    results_clean["home_score"] = pd.to_numeric(results_clean["home_score"], errors='coerce').fillna(0).astype(int)
    results_clean["away_score"] = pd.to_numeric(results_clean["away_score"], errors='coerce').fillna(0).astype(int)
    results_clean = results_clean[(results_clean["year"] >= 1872) & (results_clean["year"] <= 2025)]
    
    goals_clean = goals.copy()
    goals_clean["home_team"] = goals_clean["home_team"].str.strip().str.title()
    goals_clean["away_team"] = goals_clean["away_team"].str.strip().str.title()
    goals_clean["team"] = goals_clean["team"].str.strip().str.title()
    goals_clean["scorer"] = goals_clean["scorer"].str.strip().fillna("Unknown Player")
    goals_clean["minute"] = pd.to_numeric(goals_clean["minute"], errors='coerce')
    goals_clean = goals_clean.dropna(subset=["minute"])
    goals_clean = goals_clean[(goals_clean["minute"] >= 1) & (goals_clean["minute"] <= 120)]
    
    for col in ['own_goal', 'penalty']:
        goals_clean[col] = goals_clean[col].fillna(False)
        goals_clean[col] = goals_clean[col].astype(bool)
    
    shootouts_clean = shootouts.copy()
    shootouts_clean["home_team"] = shootouts_clean["home_team"].str.strip().str.title()
    shootouts_clean["away_team"] = shootouts_clean["away_team"].str.strip().str.title()
    shootouts_clean["winner"] = shootouts_clean["winner"].str.strip().str.title()
    
    former_clean = former.copy()
    former_clean["current"] = former_clean["current"].str.strip().str.title()
    former_clean["former"] = former_clean["former"].str.strip().str.title()
    
    for col in ['start_date', 'end_date']:
        former_clean[col] = pd.to_datetime(former_clean[col], errors='coerce')
    
    return results_clean, goals_clean, shootouts_clean, former_clean

@st.cache_data(ttl=600)
def get_country_list(results):
    home_teams = set(results["home_team"].dropna())
    away_teams = set(results["away_team"].dropna())
    countries = sorted(list(home_teams.union(away_teams)))
    valid_countries = [c for c in countries if len(c) > 1 and not c.isdigit()]
    return valid_countries

@st.cache_data(ttl=300)
def filter_country_data(results, goals, shootouts, former, country):
    country_mask = (results["home_team"] == country) | (results["away_team"] == country)
    data_negara = results[country_mask].copy()
    
    def hitung_hasil(row, negara):
        if row["home_team"] == negara:
            if row["home_score"] > row["away_score"]:
                return "Menang"
            elif row["home_score"] < row["away_score"]:
                return "Kalah"
            else:
                return "Seri"
        elif row["away_team"] == negara:
            if row["away_score"] > row["home_score"]:
                return "Menang"
            elif row["away_score"] < row["home_score"]:
                return "Kalah"
            else:
                return "Seri"

    data_negara["hasil"] = data_negara.apply(lambda row: hitung_hasil(row, country), axis=1)
    
    goal_mask = (goals["home_team"] == country) | (goals["away_team"] == country)
    goal_negara = goals[goal_mask].copy()
    
    shoot_mask = (shootouts["home_team"] == country) | (shootouts["away_team"] == country)
    shoot_negara = shootouts[shoot_mask].copy()
    
    former_mask = (former["current"].str.lower() == country.lower()) | (former["former"].str.lower() == country.lower())
    former_negara = former[former_mask].copy()
    
    return data_negara, goal_negara, shoot_negara, former_negara

@st.cache_data(ttl=300)
def calculate_additional_stats(data_negara, goal_negara, country):
    if data_negara.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    head_to_head = []
    for opponent in set(data_negara["home_team"]).union(set(data_negara["away_team"])):
        if opponent != country and pd.notna(opponent):
            matches = data_negara[
                ((data_negara["home_team"] == country) & (data_negara["away_team"] == opponent)) |
                ((data_negara["home_team"] == opponent) & (data_negara["away_team"] == country))
            ]
            if len(matches) > 0:
                wins = len(matches[matches["hasil"] == "Menang"])
                draws = len(matches[matches["hasil"] == "Seri"])
                losses = len(matches[matches["hasil"] == "Kalah"])
                total = len(matches)
                
                if wins + draws + losses == total:
                    head_to_head.append({
                        "Lawan": opponent,
                        "Menang": wins,
                        "Seri": draws,
                        "Kalah": losses,
                        "Total": total,
                        "Persentase Kemenangan": (wins / total * 100) if total > 0 else 0
                    })
    
    head_to_head_df = pd.DataFrame(head_to_head)
    
    tournament_stats = data_negara.groupby("tournament").agg({
        "hasil": lambda x: (x == "Menang").sum(),
        "date": "count"
    }).rename(columns={"hasil": "Menang", "date": "Total"}).reset_index()
    
    tournament_stats = tournament_stats[tournament_stats["Total"] > 0]
    tournament_stats["Persentase Kemenangan"] = (tournament_stats["Menang"] / tournament_stats["Total"] * 100).round(1)
    
    goal_timeline = goal_negara[goal_negara["team"] == country].copy()
    if not goal_timeline.empty:
        goal_timeline["date"] = pd.to_datetime(goal_timeline["date"], errors='coerce')
        goal_timeline = goal_timeline.dropna(subset=["date"])
        goal_timeline["year"] = goal_timeline["date"].dt.year
        goals_by_year = goal_timeline.groupby("year").size().reset_index(name="Jumlah Gol")
    else:
        goals_by_year = pd.DataFrame({"year": [], "Jumlah Gol": []})
    
    return head_to_head_df, tournament_stats, goals_by_year

@st.cache_data(ttl=300)
def calculate_advanced_stats(data_negara, goal_negara, country):
    if data_negara.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    data_negara["decade"] = (data_negara["year"] // 10) * 10
    decade_stats = data_negara.groupby("decade").agg({
        "hasil": lambda x: (x == "Menang").sum(),
        "date": "count",
        "home_score": "sum",
        "away_score": "sum"
    }).rename(columns={"hasil": "Menang", "date": "Total", "home_score": "Home_Goals", "away_score": "Away_Goals"}).reset_index()
    
    decade_stats["Win_Rate"] = (decade_stats["Menang"] / decade_stats["Total"] * 100).round(1)
    decade_stats["Goals_Per_Match"] = ((decade_stats["Home_Goals"] + decade_stats["Away_Goals"]) / decade_stats["Total"]).round(2)
    
    home_games = data_negara[data_negara["home_team"] == country]
    away_games = data_negara[data_negara["away_team"] == country]
    
    home_away_stats = pd.DataFrame({
        "Tipe": ["Kandang", "Tandang"],
        "Total_Pertandingan": [len(home_games), len(away_games)],
        "Menang": [
            len(home_games[home_games["hasil"] == "Menang"]),
            len(away_games[away_games["hasil"] == "Menang"])
        ],
        "Win_Rate": [
            (len(home_games[home_games["hasil"] == "Menang"]) / len(home_games) * 100) if len(home_games) > 0 else 0,
            (len(away_games[away_games["hasil"] == "Menang"]) / len(away_games) * 100) if len(away_games) > 0 else 0
        ]
    })
    
    goal_analysis = goal_negara[goal_negara["team"] == country]
    if not goal_analysis.empty:
        goal_types = {
            "Normal": len(goal_analysis[(goal_analysis["own_goal"] == False) & (goal_analysis["penalty"] == False)]),
            "Penalty": len(goal_analysis[goal_analysis["penalty"] == True]),
            "Own Goal": len(goal_analysis[goal_analysis["own_goal"] == True])
        }
        goal_types_df = pd.DataFrame(list(goal_types.items()), columns=["Tipe_Gol", "Jumlah"])
    else:
        goal_types_df = pd.DataFrame({"Tipe_Gol": [], "Jumlah": []})
    
    return decade_stats, home_away_stats, goal_types_df

with st.spinner("🔄 Loading data..."):
    results, goals, shootouts, former = load_and_clean_data()

st.markdown('<h1 class="main-header">Statistik Sepak Bola Dunia</h1>', unsafe_allow_html=True)

col_header1, col_header2, col_header3 = st.columns([1,2,1])
with col_header2:
    st.markdown(
        "Halo, disini kamu bisa lihat berbagai data menarik soal pertandingan internasional dari tahun 1872–2025!"
    )

st.markdown("---")

with st.sidebar:
    st.markdown("### 🌍 Pilih Negara")
    
    negara_list = get_country_list(results)
    pilih_negara = st.selectbox(
        "Pilih negara yang ingin dilihat datanya:", 
        negara_list, 
        index=negara_list.index("Indonesia") if "Indonesia" in negara_list else 0
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Filter Lanjutan")
    
    tahun_min = int(results["year"].min())
    tahun_max = int(results["year"].max())
    tahun_range = st.slider(
        "Rentang Tahun:",
        min_value=tahun_min,
        max_value=tahun_max,
        value=(tahun_min, tahun_max)
    )

data_negara, goal_negara, shoot_negara, former_negara = filter_country_data(results, goals, shootouts, former, pilih_negara)

if not data_negara.empty:
    data_negara = data_negara[
        (data_negara["year"] >= tahun_range[0]) & 
        (data_negara["year"] <= tahun_range[1])
    ]
    
    goal_negara = goal_negara[
        (goal_negara["team"] == pilih_negara) &
        (pd.to_datetime(goal_negara["date"]).dt.year >= tahun_range[0]) &
        (pd.to_datetime(goal_negara["date"]).dt.year <= tahun_range[1])
    ]

if not shoot_negara.empty:
    shoot_negara = shoot_negara[
        (pd.to_datetime(shoot_negara["date"]).dt.year >= tahun_range[0]) &
        (pd.to_datetime(shoot_negara["date"]).dt.year <= tahun_range[1])
    ]

if not data_negara.empty:
    st.markdown("### 📈 Ringkasan Singkat")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_matches = len(data_negara)
        st.metric("Total Pertandingan", f"{total_matches:,}")
    
    with col2:
        wins = len(data_negara[data_negara["hasil"] == "Menang"])
        win_rate = (wins / total_matches * 100) if total_matches > 0 else 0
        st.metric("Persentase Kemenangan", f"{win_rate:.1f}%")
    
    with col3:
        total_goals = len(goal_negara[goal_negara["team"] == pilih_negara])
        st.metric("Gol Tercatat", f"{total_goals:,}")
    
    with col4:
        first_year = data_negara["year"].min()
        last_year = data_negara["year"].max()
        st.metric("Rentang Tahun", f"{first_year}-{last_year}")

st.markdown("---")
st.markdown('<div class="section-header">📊 Performa per-dekade</div>', unsafe_allow_html=True)

if not data_negara.empty:
    decade_stats, home_away_stats, goal_types_df = calculate_advanced_stats(data_negara, goal_negara, pilih_negara)
    
    if not decade_stats.empty:
        col_decade1, col_decade2 = st.columns(2)
        
        with col_decade1:
            fig_decade1 = px.line(
                decade_stats,
                x="decade",
                y="Win_Rate",
                title=f"Persentase kemenangan {pilih_negara} per-dekade",
                markers=True,
                color_discrete_sequence=['#1f77b4']
            )
            fig_decade1.update_layout(xaxis_title="Dekade", yaxis_title="Persentase Kemenangan (%)", hovermode="x unified")
            st.plotly_chart(fig_decade1, use_container_width=True)
            st.caption("📈 Lihat perkembangan performa tim tiap dekade!")
        
        with col_decade2:
            fig_decade2 = px.bar(
                decade_stats,
                x="decade",
                y="Goals_Per_Match",
                title=f"Rata-rata Gol per Pertandingan {pilih_negara} per-dekade",
                color="Goals_Per_Match",
                color_continuous_scale="viridis"
            )
            fig_decade2.update_layout(xaxis_title="Dekade", yaxis_title="Gol per Pertandingan")
            st.plotly_chart(fig_decade2, use_container_width=True)
            st.caption("⚽ Era mana yang paling produktif dalam mencetak gol?")

st.markdown('<div class="section-header">🏠🏃 Performa Kandang vs Tandang</div>', unsafe_allow_html=True)

if not home_away_stats.empty:
    col_home1, col_home2 = st.columns(2)
    
    with col_home1:
        fig_home = px.bar(
            home_away_stats,
            x="Tipe",
            y="Win_Rate",
            title=f"Persentase Kemenangan {pilih_negara} - Kandang vs Tandang",
            color="Tipe",
            color_discrete_sequence=['#2ecc71', '#3498db']
        )
        fig_home.update_layout(yaxis_title="Persentase Kemenangan (%)", showlegend=False)
        st.plotly_chart(fig_home, use_container_width=True)
        st.caption("🔍 Tim lebih kuat main di kandang atau tandang?")
    
    with col_home2:
        fig_home2 = px.pie(
            home_away_stats,
            values="Total_Pertandingan",
            names="Tipe",
            title=f"Distribusi Pertandingan {pilih_negara} - Kandang vs Tandang",
            color_discrete_sequence=['#2ecc71', '#3498db']
        )
        st.plotly_chart(fig_home2, use_container_width=True)
        st.caption("📊 Seberapa sering tim main di kandang vs tandang?")

st.markdown('<div class="section-header">🎯 Analisis Tipe Gol</div>', unsafe_allow_html=True)

if not goal_types_df.empty and len(goal_types_df) > 0:
    col_goal1, col_goal2 = st.columns([2, 1])
    
    with col_goal1:
        fig_goal = px.pie(
            goal_types_df,
            values="Jumlah",
            names="Tipe_Gol",
            title=f"Distribusi Tipe Gol {pilih_negara}",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_goal, use_container_width=True)
    
    with col_goal2:
        st.markdown("### 📝 Breakdown Gol")
        for _, row in goal_types_df.iterrows():
            percentage = (row["Jumlah"] / goal_types_df["Jumlah"].sum() * 100) if goal_types_df["Jumlah"].sum() > 0 else 0
            st.metric(label=row["Tipe_Gol"], value=f"{row['Jumlah']}", delta=f"{percentage:.1f}%")
    
    st.caption("🎪 Lihat komposisi gol - berapa banyak dari penalty, own goal, atau gol normal?")

st.markdown('<div class="section-header">🕒 Timeline Interaktif Pertandingan</div>', unsafe_allow_html=True)

if not data_negara.empty:
    timeline_data = data_negara.nlargest(50, "date")[["date", "home_team", "away_team", "home_score", "away_score", "hasil", "tournament"]].copy()
    timeline_data["date_str"] = timeline_data["date"].dt.strftime("%Y-%m-%d")
    timeline_data["match_result"] = timeline_data.apply(
        lambda x: f"{x['home_team']} {x['home_score']}-{x['away_score']} {x['away_team']}", axis=1
    )
    
    fig_timeline = px.scatter(
        timeline_data,
        x="date",
        y="tournament",
        color="hasil",
        size_max=20,
        title=f"50 Pertandingan Terakhir {pilih_negara}",
        hover_data={"match_result": True, "date_str": True},
        color_discrete_map={"Menang": "#2ecc71", "Seri": "#f1c40f", "Kalah": "#e74c3c"}
    )
    
    fig_timeline.update_layout(xaxis_title="Tanggal", yaxis_title="Turnamen", height=400)
    st.plotly_chart(fig_timeline, use_container_width=True)
    st.caption("🕓 Timeline interaktif pertandingan terakhir. Hover untuk detail setiap pertandingan!")

st.markdown('<div class="section-header">📅 Tren Pertandingan per Tahun</div>', unsafe_allow_html=True)

if not data_negara.empty:
    tren_tahun = data_negara.groupby("year").size().reset_index(name="jumlah_pertandingan")
    
    if not tren_tahun.empty:
        fig1 = px.line(
            tren_tahun,
            x="year",
            y="jumlah_pertandingan",
            title=f"Tren Jumlah Pertandingan {pilih_negara} dari Tahun ke Tahun",
            markers=True,
            color_discrete_sequence=['#1f77b4']
        )
        fig1.update_layout(hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)
        st.caption(f"📊 Terlihat kalau {pilih_negara} punya tren pertandingan yang {'naik turun tergantung turnamen besar' if len(tren_tahun)>0 else 'belum banyak datanya nih!'}.")

st.markdown('<div class="section-header">🏆 Rekor Menang, Seri, Kalah</div>', unsafe_allow_html=True)

if not data_negara.empty:
    rekap_hasil = data_negara["hasil"].value_counts().reset_index()
    rekap_hasil.columns = ["Hasil", "Jumlah"]

    if not rekap_hasil.empty:
        fig2 = px.bar(
            rekap_hasil,
            x="Hasil",
            y="Jumlah",
            color="Hasil",
            color_discrete_sequence=["#2ecc71", "#f1c40f", "#e74c3c"],
            title=f"Rekor Hasil Pertandingan {pilih_negara}",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"⚔️ {pilih_negara} punya total {len(data_negara)} pertandingan yang tercatat di dataset ini!")

st.markdown('<div class="section-header">⚡ Pencetak Gol Terbanyak</div>', unsafe_allow_html=True)

pencetak_gol = goal_negara[goal_negara["team"] == pilih_negara]

if not pencetak_gol.empty:
    valid_scorers = pencetak_gol[pencetak_gol["scorer"] != "Unknown Player"]
    top_scorers = valid_scorers["scorer"].value_counts().head(10).reset_index()
    top_scorers.columns = ["Pemain", "Jumlah Gol"]
    
    if not top_scorers.empty:
        fig3 = px.bar(
            top_scorers,
            x="Jumlah Gol",
            y="Pemain",
            orientation="h",
            title=f"Top 10 Pencetak Gol {pilih_negara}",
            color="Jumlah Gol",
            color_continuous_scale="blues",
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("🔥 Ini dia deretan pemain yang paling sering nyetak gol buat negaranya!")

st.markdown('<div class="section-header">🤼 Head-to-Head vs Lawan Teratas</div>', unsafe_allow_html=True)

head_to_head_df, tournament_stats, goals_by_year = calculate_additional_stats(data_negara, goal_negara, pilih_negara)

if not head_to_head_df.empty:
    top_opponents = head_to_head_df.nlargest(10, "Total")
    
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name='Menang', y=top_opponents['Lawan'], x=top_opponents['Menang'], orientation='h', marker_color='#2ecc71'))
    fig4.add_trace(go.Bar(name='Seri', y=top_opponents['Lawan'], x=top_opponents['Seri'], orientation='h', marker_color='#f1c40f'))
    fig4.add_trace(go.Bar(name='Kalah', y=top_opponents['Lawan'], x=top_opponents['Kalah'], orientation='h', marker_color='#e74c3c'))
    
    fig4.update_layout(title=f"Head-to-Head {pilih_negara} vs Lawan Terbanyak", barmode='stack', height=500)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("💪 Lihat nih, rival-rival terberat dan yang paling sering ditaklukkan!")

st.markdown('<div class="section-header">🏅 Performa di Berbagai Turnamen</div>', unsafe_allow_html=True)

if not tournament_stats.empty:
    significant_tournaments = tournament_stats[tournament_stats["Total"] >= 5].nlargest(10, "Total")
    
    if not significant_tournaments.empty:
        fig5 = px.scatter(
            significant_tournaments,
            x="Total",
            y="Persentase Kemenangan",
            size="Menang",
            color="Persentase Kemenangan",
            hover_name="tournament",
            size_max=60,
            title=f"Performa {pilih_negara} di Berbagai Turnamen",
            color_continuous_scale="viridis"
        )
        fig5.update_layout(xaxis_title="Total Pertandingan", yaxis_title="Persentase Kemenangan (%)")
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("🎯 Turnamen yang paling atas adalah yang paling sering dimainkan DAN punya persentase kemenangan tertinggi!")

st.markdown('<div class="section-header">📈 Timeline Gol dari Masa ke Masa</div>', unsafe_allow_html=True)

if not goals_by_year.empty and len(goals_by_year) > 1:
    fig6 = px.area(
        goals_by_year,
        x="year",
        y="Jumlah Gol",
        title=f"Perkembangan Jumlah Gol {pilih_negara} per Tahun",
        color_discrete_sequence=["#ff6b6b"]
    )
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("🚀 Grafik ini menunjukkan era-era produktif dimana tim sering mencetak gol!")

st.markdown('<div class="section-header">🥅 Catatan Adu Penalti</div>', unsafe_allow_html=True)

shoot_display = shoot_negara[
    (shoot_negara["home_team"] == pilih_negara) | (shoot_negara["away_team"] == pilih_negara)
][["date", "home_team", "away_team", "winner"]].copy()

if not shoot_display.empty:
    shoot_display.rename(columns={"date": "Tanggal", "home_team": "Tim Tuan Rumah", "away_team": "Tim Tamu", "winner": "Pemenang"}, inplace=True)
    st.dataframe(shoot_display, hide_index=True, use_container_width=True)
    st.caption(f"🎯 {pilih_negara} terlibat di {len(shoot_display)} pertandingan yang berakhir lewat adu penalti.")
else:
    st.info("⚪ Belum ada catatan adu penalti untuk negara ini.")

st.markdown('<div class="section-header">🌍 Apakah Negara Ini Pernah Berganti Nama?</div>', unsafe_allow_html=True)

former_filtered = former[
    (former["current"].str.lower() == pilih_negara.lower()) | (former["former"].str.lower() == pilih_negara.lower())
]

if not former_filtered.empty:
    former_filtered.rename(columns={"current": "Nama Sekarang", "former": "Nama Lama", "start_date": "Mulai Digunakan", "end_date": "Selesai Digunakan"}, inplace=True)
    st.dataframe(former_filtered, hide_index=True, use_container_width=True)
    st.caption(f"🧭 Menarik! {pilih_negara} pernah mengalami perubahan nama resmi.")
else:
    st.info(f"✨ {pilih_negara} belum pernah tercatat mengalami perubahan nama resmi.")


st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns([1,2,1])
with col_footer2:
    st.markdown("<div style='text-align: center; color: #666; font-size: 0.9rem;'>Made with ❤️ using Streamlit | Data Source: Kaggle International Football Results</div>", unsafe_allow_html=True)