import os
import PIL.Image as PImage
import matplotlib.pyplot as plt
import pickle
import shutil
import json

folder = "../Files"
file_list = list()

if os.path.exists(folder):
    if os.path.isdir(folder):
        list_of_files = os.scandir(folder)
        for file in list_of_files:
            file_path = os.path.join(folder, file.name)
            if os.path.isfile(file_path):
                print(file_path, "exists")
                file_list.append(file.name)
            filename, file_extension = os.path.splitext(file_path)
            if file_extension == ".txt" or file_extension == ".json":
                with open(file_path, 'r') as file_text:
                    for line in file_text:
                        print(line)
                        alt_line = line.replace('\n', '')
                        print(alt_line)
            if file_extension == ".jpg" or file_extension == ".png" or file_extension == ".tif":
                image = PImage.open(file_path)
                plt.imshow(image)
                plt.show()

# pickle
with open("list_of_files.data", "wb") as fp:  # Pickling
    pickle.dump(file_list, fp)
list_of_files = ""
with open("list_of_files.data", "rb") as fp:
    print("Loading list_of_files")
    list_of_files = pickle.load(fp)

os.remove("list_of_files.data")

new_folder = os.path.join(folder, 'Copies')
if not os.path.exists(new_folder):
    os.mkdir(new_folder)

for file in list_of_files:
    shutil.copyfile(os.path.join(folder, file),
                    os.path.join(new_folder, file))

list_of_new_files = os.scandir(new_folder)

for file in list_of_new_files:
    os.remove(os.path.join(new_folder, file.name))
os.rmdir(new_folder)

var = 1
with open(os.path.join(folder, "var_int.json"), "w") as fp:
    json.dump(var, fp)
with open(os.path.join(folder, "var_int.json"), "r") as data_file:
    var2 = json.load(data_file)
print(var2)

var = 0.1
with open(os.path.join(folder, "var_float.json"), "w") as fp:
    json.dump(var, fp)
with open(os.path.join(folder, "var_float.json"), "r") as data_file:
    var2 = json.load(data_file)
print(var2)

var = "hello"
with open(os.path.join(folder, "var_string.json"), "w") as fp:
    json.dump(var, fp)
with open(os.path.join(folder, "var_string.json"), "r") as data_file:
    var2 = json.load(data_file)
print(var2)

var = (1, 3, 5, 7, 9)
with open(os.path.join(folder, "var_tuple.json"), "w") as fp:
    json.dump(var, fp)
with open(os.path.join(folder, "var_tuple.json"), "r") as data_file:
    var2 = json.load(data_file)
print(var2)

var = [2, 4, 6, 8, 10]
with open(os.path.join(folder, "var_list.json"), "w") as fp:
    json.dump(var, fp)
with open(os.path.join(folder, "var_list.json"), "r") as data_file:
    var2 = json.load(data_file)
print(var2)

var = {"BAIT": "Brito Artificial Intelligence Team",
            "MEEC": "Master in Electronics Engineering and Computers",
            "EST": "Escola Superior de Tecnologia",
            "IPCA": "Instituto Politécnico do Cávado e do Ave"}
with open(os.path.join(folder, "var_dict.json"), "w") as fp:
    json.dump(var, fp)
with open(os.path.join(folder, "var_dict.json"), "r") as data_file:
    var2 = json.load(data_file)
print(var2)

os.remove(os.path.join(folder, "var_int.json"))
os.remove(os.path.join(folder, "var_float.json"))
os.remove(os.path.join(folder, "var_string.json"))
os.remove(os.path.join(folder, "var_tuple.json"))
os.remove(os.path.join(folder, "var_list.json"))
os.remove(os.path.join(folder, "var_dict.json"))
