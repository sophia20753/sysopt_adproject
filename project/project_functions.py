# project_functions.py

# import relevant libraries
import os
import pandas as pd

def save_data_to_csv(foldername, filename, data):
    '''Saves data from dict to a .csv file'''
    df = pd.DataFrame(data)
    csv_file_path = os.path.join(foldername, filename)
    df.to_csv(csv_file_path, index=False)
    print(f"Saved {filename}")