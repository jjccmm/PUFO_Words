import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
#from datasets import load_dataset
import torch
import numpy as np
import time
import re


def get_episode_links_old():
    """Gets the episode links from the Pufopedia website.
    """
    url = "https://www.pufopedia.info/wiki/Kategorie:Folgen"
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.content, "html.parser")

    table = soup.find("table", class_="wikitable")

    titles = []
    download_links = []
    numbers = []
    
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) >= 5:
            title = cols[0].text.strip()
            number = int(title.split(" ")[0][3:])
            download_td = cols[4]
            link_tag = download_td.find("a", href=True)
            if link_tag:
                link = link_tag['href']
                titles.append(title)
                download_links.append(link)
                numbers.append(number)

    df_link = pd.DataFrame({
        "episode_number": numbers,
        "episode_title": titles,
        "link": download_links
    })
    return df_link




def get_episode_links():   
    url = "https://feeds.acast.com/public/shows/podcast-ufo"
    response = requests.get(url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, "xml")
    
    items = soup.find_all("item")
    
    episodes = []
    for item in items:
        title = item.find("title").text.strip()
        enclosure = item.find("enclosure")
        media_url = enclosure["url"] if enclosure else None

        # Episoden-Nummer extrahieren
        match = re.search(r"UFO(\d+)", title)
        episode_number = int(match.group(1)) if match else None

        if episode_number:
            episodes.append({
                "number": episode_number,
                "title": title,
                "link": media_url
            })

    # In DataFrame
    df = pd.DataFrame(episodes)

    # Nach Episodennummer sortieren, kleinste zuerst
    df = df.sort_values(by="number").reset_index(drop=True)
    
    return df


def load_episodes_df():
    df_episodes = pd.read_csv("episodes.csv", header=0)
    return df_episodes


def add_new_episodes(df_episodes):
    df_link = get_episode_links()
    new_episodes = df_link[~df_link['number'].isin(df_episodes['number'])]
    df_episodes = pd.concat([df_episodes, new_episodes], ignore_index=True)
    df_episodes.to_csv("episodes.csv", index=False)
    print(df_episodes)
    return df_episodes
    

def download_episode(episode_number, episode_link):
    print(f"Downloading {episode_number} from {episode_link}")

    download_folder = "downloads"
    os.makedirs(download_folder, exist_ok=True)
    
    filename = f"episode_{episode_number}.mp3"
    filepath = os.path.join(download_folder, filename)
    
    try:
        response = requests.get(episode_link, stream=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"✅ Heruntergeladen: {filename}")
            return "download_done"
        else:
            print(f"❌ Fehler beim Download: {episode_link} (Status {response.status_code})")
            return "download_error"
    except Exception as e:
        print(f"⚠️ Ausnahme bei {episode_link}: {e}")
        return "download_error"
    
    
def delete_download_folder():
    download_folder = "downloads"
    if os.path.exists(download_folder):
        for filename in os.listdir(download_folder):
            file_path = os.path.join(download_folder, filename)
            try:
                os.remove(file_path)
                print(f"✅ Gelöscht: {file_path}")
            except Exception as e:
                print(f"⚠️ Fehler beim Löschen von {file_path}: {e}")
        os.rmdir(download_folder)
        print(f"✅ Ordner gelöscht: {download_folder}")
    else:
        print("❌ Download-Ordner existiert nicht.")
    
    
def process_episode(pipe, episode_number):
    print(f"Processing episode {episode_number}...")
    start_time = time.time()

    prediction = pipe(f"downloads/episode_{episode_number}.mp3")["chunks"]

    end_time = time.time()
    print(f"Time taken: {(end_time - start_time)/60} minutes")

    with open(f"text/episode_{episode_number}.txt", "w", encoding="utf-8") as f:
        f.write(str(prediction))
        
    if prediction is not None:
        print(f"✅ Episode {episode_number} erfolgreich verarbeitet.")
        return "done"
    else:
        print(f"❌ Fehler bei der Verarbeitung von Episode {episode_number}.")
        return "processing_error"
    
    
def prepare_whisper_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "primeline/whisper-large-v3-turbo-german"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=8,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
    )
    return pipe    
    
    
    
    
if __name__ == "__main__":
    check_for_new_episodes = True
    
    df_episodes = load_episodes_df()
    if check_for_new_episodes:
        df_episodes = add_new_episodes(df_episodes)

    
    pipe = prepare_whisper_model()
    for index, row in df_episodes.iterrows():
        episode_link = row['link']
        episode_name = row['title']
        episode_state = row['state']
        episode_number = row['number']    
        
        print(f"Episode: {episode_name}")

        if episode_state == 'done':
            print(f"Episode {episode_name} already processed.")
            continue
    
        if episode_state == 'skip':
            print(f"Episode {episode_name} will be skipped.")
            continue
        
        state = download_episode(episode_number, episode_link)
        if state == "download_error":
            df_episodes.at[index, 'state'] = 'skip'
        else:
            state = process_episode(pipe, episode_number)
            df_episodes.at[index, 'state'] = state
        
        df_episodes.to_csv("episodes.csv", index=False)
        delete_download_folder()
    