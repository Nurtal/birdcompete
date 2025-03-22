import pandas as pd
import random




def generate_random_submission_file():
    """Generate random prediction, just to test the submission process"""

    # load manifest
    sample_list = list(pd.read_csv("data/submission_manifest.csv")['SAMPLE'])

    # load species
    species_list = list(pd.read_csv('data/sample_submission.csv').keys())[1:]

    # generate random values
    data = []
    for sap in sample_list:
        vector = {'row_id':sap}
        for spec in species_list:
            vector[spec] = random.uniform(0,1)
        data.append(vector)

    # craft dataset
    df = pd.DataFrame(data)
    print(df)
    






def craft(result_folder, sample_manifest):
    """Craft a formated submission file from content of a result folder and sample manofest"""

    


if __name__ == "__main__":

    # params
    result_folder = "/tmp/zog"
    sample_manifest = "data/submission_manifest.csv"

    # craft()
    generate_random_submission_file()
