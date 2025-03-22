import os
import shutil
import glob
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm



def craft_train_db(sample_duration:int, original_data_folder:str, output_train_folder:str) -> None:
    """Craft a train dataset and a label file associated. Each audio sample is chunk into small part
    of provided duration.

    Args:
        - sample_duration (int) : lenght of audio sample for the train dataset (miliseconds)
        - original_data_folder (str) : path to the original data folder
        - output_train_folder (str) : path to the folder generated for the train dataset
    
    """

    # test if original data folder exist
    if not os.path.isdir(original_data_folder):
        print(f"[!] Can't find {original_data_folder}")
        return -1

    # init output folder
    if os.path.isdir(output_train_folder):
        shutil.rmtree(output_train_folder)
    os.mkdir(output_train_folder)

    # init labels
    labels = []

    # transfert
    cmpt = 0
    cmpt_chunk = 0
    for folder in tqdm(glob.glob(f"{original_data_folder}/*"), desc="Crafting dataset ..."):

        output_folder = folder.split("/")[-1]
        output_folder = f"{output_train_folder}/{output_folder}"
        os.mkdir(output_folder)
        label = folder.split("/")[-1]

        for audio_file in glob.glob(f"{folder}/*.ogg"):

            # Load audio
            audio = AudioSegment.from_ogg(audio_file)
            duree_totale = len(audio)  # Durée totale en millisecondes
            cmpt+=1

            # Découpage
            for i, debut in enumerate(range(0, duree_totale, sample_duration)):
                fin = min(debut + sample_duration, duree_totale)
                chunk = audio[debut:fin]

                # get output filename
                output_file_name = f"{output_folder}/{audio_file.split('/')[-1].split('.')[0]}_chunk_{i}.ogg"

                # Vérifier que le chunk est bien de la taille requise
                if len(chunk) == sample_duration:
                    chunk.export(output_file_name, format="ogg")
                    labels.append({'ID':output_file_name, 'LABEL':label})

                # udpate cmpt
                cmpt_chunk+=1

    # write label file
    df = pd.DataFrame(labels)
    df.to_csv(f"{output_train_folder}/labels.csv")

    # display infos
    print(f"[+] {cmpt} audio files treated")
    print(f"[+] {cmpt_chunk} audio chunk created")


if __name__ == "__main__":

    craft_train_db(16000, "data/train_audio", "data/zog1sec")
