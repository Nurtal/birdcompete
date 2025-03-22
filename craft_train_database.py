import os
import shutil
import glob
from pydub import AudioSegment




def craft_train_db(sample_duration:int, original_data_folder:str, output_train_folder:str):
    """ """

    # test if original data folder exist
    if not os.path.isdir(original_data_folder):
        print(f"[!] Can't find {original_data_folder}")
        return -1

    # init output folder
    if os.path.isdir(output_train_folder):
        shutil.rmtree(output_train_folder)
    os.mkdir(output_train_folder)

    # transfert
    cmpt = 0
    cmpt_chunk = 0
    for folder in glob.glob(f"{original_data_folder}/*"):

        output_folder = folder.split("/")[-1]
        output_folder = f"{output_train_folder}/{output_folder}"
        os.mkdir(output_folder)

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

                # udpate cmpt
                cmpt_chunk+=1

    # display infos
    print(f"[+] {cmpt} audio files treated")
    print(f"[+] {cmpt_chunk} audio chunk created")


    


            
        


if __name__ == "__main__":

    craft_train_db(16000, "data/original_dataset", "/tmp/zog")
