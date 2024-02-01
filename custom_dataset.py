import os
from torch.utils.data import Dataset
import pandas as pd

class custom_dataset(Dataset):
    def __init__(self, data_folder, transform=None):
        super().__init__()

        # the file hierarchy takes a path to a data folder and has a
        # loop to go through the files in the subfolder Focused and to go through the subfolder Unfocused

        self.transform = transform
        self.values = []
        self.labels = []
        true_value = 1
        false_value = 0

        # iterate through subdirectories of data_folder,
        for subfolder in os.listdir(data_folder):
            subfolder_path = os.path.join(data_folder, subfolder)
            # iterate through files in subdirectory
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                values = []
                labels = []
                # Check if subfolder is named "Focused" or "Unfocused"
                if subfolder == "Focused":
                    values, labels = self.getDataFromFile(file_path, true_value, transform)
                elif subfolder == "Unfocused":
                    values, labels = self.getDataFromFile(file_path, false_value, transform)

                self.values += values
                self.labels += labels

        print("Dataset loaded")


    @staticmethod
    def getDataFromFile(file, is_focused, transform):

        # takes txt imput and converts to csv file
        df = pd.read_csv(file, skiprows=4)

        # Clean up column names by stripping leading/trailing spaces and replacing spaces with underscores
        df.columns = df.columns.str.strip().str.replace(' ', '_')

        # Drop columns with all zero or NaN values
        df = df.dropna(axis=1, how='all')

        df_grouped = df.groupby(df.index // 4).agg({
            'Sample_Index': 'first',
            'EXG_Channel_0': 'mean',
            'EXG_Channel_1': 'mean',
            'EXG_Channel_2': 'mean',
            'EXG_Channel_3': 'mean',
            'Accel_Channel_0': 'first',
            'Accel_Channel_1': 'first',
            'Accel_Channel_2': 'first',
            'Timestamp': 'first',
            'Timestamp_(Formatted)': 'first'
        })

        df_grouped.to_csv('outputClean.csv', index=False)

        # Read in the cleaned csv file
        df = pd.read_csv('outputClean.csv')
        #drop first column
        df = df.drop(df.columns[0], axis=1)
        #drop last two
        df = df.drop(df.columns[-1], axis=1)
        df = df.drop(df.columns[-1], axis=1)
        #drop first row
        df = df.drop(df.index[0])

        final_values = []
        final_labels = []

        #iterate through each line in clean csv
        for index, line in df.iterrows():
            #append to values an array of all sensor values for each sample
            values = line.values
            #convert ndarray vallues to an array of floats
            # values = values.astype(float)

            final_values += [values]

            # append to focused a boolean value indicating whether the user was focused or not
            final_labels.append(is_focused)


        return final_values, final_labels

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        value = self.values[index]
        focused = self.labels[index]
        if self.transform:
            value = self.transform(value)
        return value, focused
