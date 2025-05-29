import os
import ast
import csv
import spacy
import pandas as pd
from collections import Counter
import numpy as np
import json

# spaCy-Modell laden
nlp = spacy.load("de_core_news_lg")

TEXT_FOLDER = "text"
current_dir = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(current_dir, '..', 'word_counts.csv')
JSON_FILE = os.path.join(current_dir, '..', 'episode_stats.json')


# Lemma-Stoppwort-Infos speichern
lemma_info = {}

# Textverarbeitung mit Lemma, Kleinbuchstaben
def process_text(text):
    tokens = nlp(text)
    lemmas = []
    for token in tokens:
        lemma = token.lemma_.lower()
        if not lemma.isalpha():
            continue
        lemmas.append(lemma)
        if lemma not in lemma_info:
            lemma_info[lemma] = token.is_stop
    return lemmas

# CSV laden, falls vorhanden
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    df.set_index("word", inplace=True)
    existing_episodes = [int(col) for col in df.columns if col != "is_stop"]
else:
    df = pd.DataFrame()
    existing_episodes = []

# Neue Episoden verarbeiten
for dateiname in os.listdir(TEXT_FOLDER):
    if dateiname.startswith("episode_") and dateiname.endswith(".txt"):
        print(f"Verarbeite {dateiname}...")
        episode_nummer = int(dateiname.split("_")[1].split(".")[0])
        if episode_nummer in existing_episodes:
            print(f"Episode {episode_nummer} bereits vorhanden")
            continue  # Schon vorhanden

        pfad = os.path.join(TEXT_FOLDER, dateiname)
        with open(pfad, "r", encoding="utf-8") as f:
            try:
                daten = ast.literal_eval(f.read())
            except Exception as e:
                print(f"Fehler in {dateiname}: {e}")
                continue

        # Alle Texte analysieren
        alle_lemmas = []
        for eintrag in daten:
            alle_lemmas.extend(process_text(eintrag.get("text", "")))

        zähler = Counter(alle_lemmas)

        # Neue Spalte für diese Episode
        neue_spalte = pd.Series(zähler, name=episode_nummer)

        # In das DataFrame einfügen
        df = df.join(neue_spalte, how="outer")
        print(f"Episode {episode_nummer} hinzugefügt")

# Fill missing cells
df.fillna(0, inplace=True)
df = df.astype({col: 'int' for col in df.columns if col != 'is_stop'})

# Update or add the is_stop column
df["is_stop"] = df.index.map(lambda x: lemma_info.get(x, False))

# Sort columns: is_stop + episodes (numerically)
spalten = ["is_stop"] + sorted([c for c in df.columns if c != "is_stop"], key=int)
df = df[spalten]

# Sort rows alphabetically by word
df.sort_index(inplace=True)

# Reset the index to make 'word' a normal column
df.reset_index(inplace=True)
# Rename the 'index' column to 'word'
df.rename(columns={"index": "word"}, inplace=True)
df.replace(0, np.nan, inplace=True)
# Save to CSV without including the index column
df.to_csv(CSV_FILE, index=False)
print(f"CSV successfully saved: {CSV_FILE}")



# CSV laden
df = pd.read_csv(CSV_FILE)

# Episoden-Spalten identifizieren (nur numerische Spalten)
episode_cols = sorted([col for col in df.columns if col not in ['word', 'is_stop']], key=int)

# Gesamtstatistik
gesamt_woerter = int(df[episode_cols].sum().sum())
gesamt_vocab = int((df[episode_cols].sum(axis=1) > 0).sum())
anzahl_episoden = len(episode_cols)

# Wortliste pro Zeile
df["word"] = df["word"].astype(str)

# Pro-Episode-Statistik
bisherige_woerter = set()
episode_stats = []

for col in episode_cols:
    episode_num = int(col)
    counts = df[col]
    
    gesprochene_woerter = int(counts.sum())
    verschiedene_woerter = int((counts > 0).sum())
    
    aktuelle_woerter = set(df.loc[counts > 0, "word"])
    neue_woerter = aktuelle_woerter - bisherige_woerter
    neue_woerter_anzahl = len(neue_woerter)

    bisherige_woerter.update(aktuelle_woerter)

    episode_stats.append({
        "episode": episode_num,
        "total_words": gesprochene_woerter,
        "unique_words": verschiedene_woerter,
        "new_words": neue_woerter_anzahl
    })

# Komplettes Ergebnis
result = {
    "total_episodes": anzahl_episoden,
    "total_words": gesamt_woerter,
    "total_unique_words": gesamt_vocab,
    "episodes": sorted(episode_stats, key=lambda x: x["episode"])
}

# Speichern als JSON
with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Statistik gespeichert als: episode_stats.json")