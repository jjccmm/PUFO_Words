import pandas as pd
import streamlit as st
import plotly.graph_objs as go


st.set_page_config(layout="wide")

# 📥 CSV laden und umstrukturieren
@st.cache_data
def load_data():
    df_raw = pd.read_csv("word_counts.csv", index_col=0)
    df = df_raw.T  # Episoden = Zeilen
    df.index.name = "Episode"
    df.reset_index(inplace=True)
    df["Episode"] = df["Episode"].astype(int)
    return df, df_raw

df, df_raw = load_data()


st.title("🎙️ Die Das-Podcast-Ufo Podcast-Wortanalyse")

# 🎯 Wörter-Auswahl
word_columns = df.columns.drop("Episode")
selected_words = st.multiselect("🔍 Wähle Wörter", word_columns, default=['eimer', 'geld', 'münzen', 'cent'])

if selected_words:
    # 📈 Häufigkeit über Episoden (Stacked Line Plot)
    st.subheader("📊 Häufigkeit der gewählten Wörter über alle Episoden")
    fig_line = go.Figure()
    for word in selected_words:
        fig_line.add_trace(go.Scatter(
            x=df["Episode"], y=df[word], mode="lines", stackgroup="one", name=word
        ))
    fig_line.update_layout(
        xaxis_title="Episode",
        yaxis_title="Anzahl",
        title="Stacked Line Plot",
        width=1000,
        height=400
    )
    st.plotly_chart(fig_line, use_container_width=False)

    # 🏆 Top 10 Episoden mit höchster Wortanzahl
    st.subheader("🏅 Top 10 Episoden mit häufigster Verwendung ausgewählter Wörter")

    df["total_selected"] = df[selected_words].sum(axis=1)

    # Top 10 nach Häufigkeit
    top10 = df.sort_values("total_selected", ascending=False).head(10)

    # Episoden als Strings für korrektes Y-Achsen-Labeling
    top10["Episode_str"] = "Episode " + top10["Episode"].astype(str)

    # Balken umgekehrt sortieren (damit häufigste oben steht)
    top10_sorted = top10.sort_values("total_selected", ascending=True)

    fig_bar = go.Figure()
    for word in selected_words:
        fig_bar.add_trace(go.Bar(
            x=top10_sorted[word],
            y=top10_sorted["Episode_str"],
            name=word,
            orientation="h"
        ))

    fig_bar.update_layout(
        barmode="stack",
        xaxis_title="Anzahl",
        yaxis_title="Episode (nur Top 10)",
        title="Top 10 Episoden (Stacked Bar Chart)",
        width=1000,
        height=500,
        yaxis=dict(type="category")  # ⬅️ Y-Achse kategorisch
    )

    st.plotly_chart(fig_bar, use_container_width=False)

else:
    st.info("⬆ Bitte wähle oben ein oder mehrere Wörter.")

# 📊 Allgemeine Statistik
st.header("📋 Allgemeine Statistik")

# 1. Gesamtanzahl Episoden
total_episodes = len(df)
# 2. Gesamtanzahl Wörter
total_words = df[word_columns].sum().sum()
# 3. Anzahl unterschiedlicher Wörter
unique_words_total = len(word_columns)

col1, col2, col3 = st.columns(3)
col1.metric("🎧 Anzahl Episoden", total_episodes)
col2.metric("🗣️ Gesprochene Wörter insgesamt", f"{total_words:,}".replace(',', '.'))
col3.metric("🔤 Verschiedene Wörter insgesamt", f"{unique_words_total:,}".replace(',', '.'))

# Wörter pro Episode
st.subheader("📈 Wörter pro Episode")
words_per_episode = df[word_columns].sum(axis=1)
fig_total = go.Figure(go.Scatter(x=df["Episode"], y=words_per_episode, mode="lines", name="Wörter pro Episode"))
fig_total.update_layout(xaxis_title="Episode", yaxis_title="Wörter", width=1000, height=300)
st.plotly_chart(fig_total, use_container_width=False)

# Verschiedene Wörter pro Episode
st.subheader("🔠 Verschiedene Wörter pro Episode")
unique_words_per_episode = (df[word_columns] > 0).sum(axis=1)
fig_unique = go.Figure(go.Scatter(x=df["Episode"], y=unique_words_per_episode, mode="lines", name="Einzigartige Wörter"))
fig_unique.update_layout(xaxis_title="Episode", yaxis_title="Anzahl verschiedener Wörter", width=1000, height=300)
st.plotly_chart(fig_unique, use_container_width=False)

# Neue Wörter pro Episode
st.subheader("🆕 Neue Wörter pro Episode")
seen_words = set()
new_words_per_episode = []

for _, row in df[word_columns].iterrows():
    current_words = set(row[row > 0].index)
    new_words = current_words - seen_words
    new_words_per_episode.append(len(new_words))
    seen_words.update(current_words)

fig_new = go.Figure(go.Scatter(
    x=df["Episode"][1:], y=new_words_per_episode[1:], mode="lines", name="Neue Wörter"
))
fig_new.update_layout(
    xaxis_title="Episode",
    yaxis_title="Neue Wörter",
    yaxis_type="log",
    title="Neue Wörter pro Episode (logarithmisch)",
    width=1000,
    height=300
)
st.plotly_chart(fig_new, use_container_width=False)
