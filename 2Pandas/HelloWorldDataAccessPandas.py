import os
import numpy as np
import pandas as pd

bad_columns_to_drop = ["X2=Equity/liabilities",
                       "X7=Current Liabilities /Inventory",
                       "X26=Financing Charge / Sales"]
useless_columns = ["Unnamed: 0",
                   "NACE Rev. 2, core code (4 digits)"]
filename = "company_info.xlsx"
folder = "../Files"
file_path = os.path.join(folder, filename)
df1 = pd.read_excel(file_path, engine="openpyxl", sheet_name=0)
df2 = pd.read_excel(file_path, engine="openpyxl", sheet_name=1)

df = pd.concat([df1, df2])

print("HEAD:{}".format(df.head()))
print("DTYPES:{}".format(df.dtypes))
print("DESCRIBE:{}".format(df.describe(include="all")))
print("INFO:{}".format(df.info))
print("Columns:{}".format(df.columns))
print("Shape:{}".format(df.shape))

df.drop(columns=useless_columns, inplace=True)
df.drop(columns=bad_columns_to_drop, inplace=True)

df.replace("ES", 0, inplace=True)
df.replace("PT", 1, inplace=True)

print("HEAD:{}".format(df.head()))

df.dropna(axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

for column in df.columns:
    if column == "Country ISO code":
        df[column] = df[column].astype("int")

Y = np.array(df["Situation"])
df.drop("Situation", axis=1, inplace=True)
X = np.array(df, dtype="float64")
