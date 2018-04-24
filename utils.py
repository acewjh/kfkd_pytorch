import pandas as pd
import numpy as np
import csv

def load_data(path):
    #load imgs
    df = pd.read_csv(path)
    key = df.keys()[-1]
    imgs = df[key]
    imgs = [i.split(" ") for i in imgs]
    imgs = np.array([[int(j) for j in i] for i in imgs])
    #load coordinates
    crds = np.array(df.iloc[:, :-1].as_matrix())
    #generate coordinate masks
    masks = np.ones(crds.shape)
    masks[np.isnan(crds)] = 0
    masks.astype(int)
    #fill null coordinates with 0
    crds[np.isnan(crds)] = 0
    return imgs, crds, masks

def load_img(path):
    df = pd.read_csv(path)
    key = df.keys()[-1]
    imgs = df[key]
    imgs = [i.split(" ") for i in imgs]
    imgs = np.array([[int(j) for j in i] for i in imgs])
    return imgs

def save_answers(pred_crds, table_path, data_path, save_path):
    data_df = pd.read_csv(data_path)
    keys = data_df.keys()
    key_dict = {}
    for i, key in enumerate(keys):
        key_dict[key] = i
    table_df = pd.read_csv(table_path)
    img_indx = [i - 1 for i in table_df['ImageId']]
    crd_indx = [key_dict[i] for i in table_df['FeatureName']]
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(img_indx)):
            crds = pred_crds[img_indx[i], crd_indx[i]]
            if crds < 0:
                crds = 0.0
            elif crds >96:
                crds = 96.0
            row = [i+1, crds]
            writer.writerow(row)
    csvfile.close()


