
# coding: utf-8

# Check [pytesseract](https://pypi.org/project/pytesseract/).



from pytesseract import *
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


def img_df3(img):
    out = image_to_data(img)

    with open('tmp', "w") as f:
        f.write(out)

    # df = pd.read_csv('tmp.csv', sep=',', engine="python")
    df = pd.DataFrame.from_csv('tmp', sep='\t')

    df = df[df.text.notnull()].reset_index(drop=True)
    os.remove('tmp')

    # assume all float numbers are prices, select them out
    tmp = df.text.apply(lambda x: bool(re.match(r"\d+\.\d{2}", x)), 1)
    df1 = df[tmp].reset_index()
    df1 = df1.rename(columns={'index': 'old_index'})

    # select only valid prices:
    def check_neighbor_exist(i):
        """ return whether there is another float number around it, this can be done by old_index """
        cur_old_index = df1.loc[i, "old_index"]
        prev_old_index = df1.loc[i - 1, "old_index"] if i != 0 else -1
        suc_old_index = df1.loc[i + 1, "old_index"] if i != len(df1) - 1 else -1
        prev = cur_old_index - prev_old_index
        suc = suc_old_index - cur_old_index
        if prev == 1 or suc == 1:
            return True
        else:
            return False

    valid_ids = [i for i in df1.index if check_neighbor_exist(i)]
    df1 = df1.iloc[valid_ids]
    df1['begin'] = 0
    #print(df1.head())

    # select out all wine No., assuming they are always the first word on the same line
    check = df.text.apply(lambda x: bool(re.match(r"[0-9]+$", x)), 1)
    df2 = df[check].reset_index()
    df2 = df2.rename(columns={'index': 'old_index'})

    # print(df2.head(10))
    def check_first_word(i):
        """ return whether it is the first word on the same line """
        return df2.loc[i, "word_num"] == 1

    valid_ids = [i for i in df2.index if check_first_word(i)]
    df2 = df2.iloc[valid_ids]
    df2['begin'] = 1

    df3 = pd.concat([df1, df2]).sort_values('old_index')

    return df, df3


#filename = os.path.join(os.getcwd(), '../sample1/UCD_Lehmann_0006.jpg')
def table_reader(filename):
    img = cv2.imread(filename)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.Laplacian(img, cv2.CV_64F)

    _, df3 = img_df3(img)
    top, bottom = max(df3.top.min() - 200, 0), df3.top.max() + 300
    left, right = max(df3.left.min() - 200, 0), df3.left.max() + 300
    df, df3 = img_df3(img[top:bottom, left:right])

    res = []
    for i in range(len(df3) - 1):
        # for i in range(5):
        if df3.iloc[i].begin == 1 and (i + 1 == len(df3) or df3.iloc[i + 1].begin == 0):
            # print()
            # print("start at:", df3.iloc[i]['old_index'], df3.iloc[i].text)
            # tmp = [None] * 6  # [No.,name, price1, price2, price3, description]
            # tmp[0] = df3.iloc[i].text
            tmp = []
            tmp.append(df3.iloc[i].text)
            start = i

            # prices:
            prices = []
            end = i + 1
            while end < len(df3) - 1 and df3.iloc[end].begin == 0:
                # tmp[end - start + 1] = df3.iloc[end].text
                prices.append(df3.iloc[end].text)
                end += 1
            # print("end at:", df3.iloc[end-1]['old_index'])

            # name:
            s = ''
            # print(int(df3.loc[i,'old_index']))
            for k in range(df3.iloc[i]['old_index'] + 1, df3.iloc[i + 1]['old_index']):
                # print(k, df.iloc[k].text)
                s += ' ' + df.iloc[k].text
            # print(s)
            s = s.rstrip(' .')
            # tmp[1] = s
            tmp.append(s)

            # description:
            s = ''
            for k in range(df3.iloc[end - 1]['old_index'] + 1, df3.iloc[end]['old_index']):
                s += ' ' + df.iloc[k].text
            # print(s)
            s = s.rstrip(' .')
            # tmp[-1] = s
            tmp.append(s)

            # res.append(tmp)
            res.append(tmp + prices)

    L = max([len(s) for s in res])
    for s in res:
        s += [''] * (L - len(s))

    #pd.set_option('display.max_columns', None)
    res_df = pd.DataFrame(np.array(res))
    #res_df.head(100)
    # res_df.to_csv(filename[:-3] + "csv")
    # return res_df

    #res_df = pd.DataFrame(np.array(res), columns=["No.", "Name", "Price1", "Price2", "Price3", "Description"])
    #res_df.head()
    res_df.to_csv(filename[:-3] + "csv")
    return res_df

