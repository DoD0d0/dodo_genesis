# import pickle

# # Replace 'path_to_your_file.pkl' with the actual path to your PKL file
# file_path = 'C:\\Users\\Liamb\\SynologyDrive\\TUM\\3_Semester\\dodo_alive\\dodo_genesis\\logs\\dodo-walking-new-004-curr005\\cfgs.pkl'

# # Open the file in binary mode and load the data
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # Now 'data' contains the deserialized Python object
# print(data)   


import pickle
from rich import print
from rich.pretty import Pretty

file_path = "C:/Users/Liamb/SynologyDrive/TUM/3_Semester/dodo_alive/dodo_genesis/logs/dodo-walking-new-004-curr005/cfgs.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

print(Pretty(data, expand_all=False))
