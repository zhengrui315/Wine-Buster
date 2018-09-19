
# coding: utf-8

# Check [pytesseract](https://pypi.org/project/pytesseract/).

# In[13]:


from pytesseract import *
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

#filename = os.path.join(os.getcwd(), '../sample1/UCD_Lehmann_0006.jpg')
def table_reader(filename):
    img = cv2.imread(filename)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.Laplacian(img, cv2.CV_64F)
    out = image_to_data(img1)

    csv_file = out.replace('\t', ',')
    with open('tmp.csv', "w") as f:
        f.write(csv_file)

    df = pd.read_csv('tmp.csv', sep=',', error_bad_lines=False, engine="python")
    df = df[df.text.notnull()].reset_index(drop=True)

    df1 = df.text.apply(lambda x: bool(re.match(r"\d+\.\d{2}", x)), 1)
    df1 = df[df1].reset_index()
    df1 = df1.rename(columns={'index': 'old_index'})
    df1['begin'] = 0
    print(df1.head())
    df2 = df.text.apply(lambda x: bool(re.match(r"[0-9]+$", x)), 1)
    df2 = df[df2].reset_index()
    df2 = df2.rename(columns={'index': 'old_index'})
    df2['begin'] = 1

    df3 = pd.concat([df1, df2]).sort_values('old_index')
    # df3.head(20)


    res = []
    for i in range(len(df3) - 1):
        # for i in range(5):
        if df3.iloc[i].begin == 1 and (i + 1 == len(df3) or df3.iloc[i + 1].begin == 0):
            # print()
            # print("start at:", df3.iloc[i]['old_index'], df3.iloc[i].text)
            tmp = [None] * 6  # [No.,name, price1, price2, price3, description]
            tmp[0] = df3.iloc[i].text
            start = i

            # prices:
            end = i + 1
            while end < len(df3) and df3.iloc[end].begin == 0:
                tmp[end - start + 1] = df3.iloc[end].text
                end += 1
            # print("end at:", df3.iloc[end-1]['old_index'])

            # name:
            s = ''
            for k in range(int(df3.iloc[i]['old_index']) + 1, int(df3.iloc[i + 1]['old_index'])):
                #print(k, df.iloc[k].text)
                s += ' ' + df.iloc[k].text
            # print(s)
            s = s.rstrip(' .')
            tmp[1] = s

            # description
            s = ''
            for k in range(int(df3.iloc[end - 1]['old_index']) + 1, int(df3.iloc[end]['old_index'])):
                s += ' ' + df.iloc[k].text
            # print(s)
            s = s.rstrip(' .')
            tmp[-1] = s

            res.append(tmp)

    res_df = pd.DataFrame(np.array(res), columns=["No.", "Name", "Price1", "Price2", "Price3", "Description"])
    res_df.head()
    res_df.to_csv(filename[:-3] + "csv")
    return res_df

