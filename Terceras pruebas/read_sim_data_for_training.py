# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:12:11 2025

@author: egarate
"""

import pandas as pd
import numpy as np
import os

def read_simulations_nad_get_XY(doe_path, data_raw_path, idxToSave=[2500, 4000], exp_id = None, sep = ";"):
    
    data_raw_path = os.path.normpath(data_raw_path)
    
    # doe_df = pd.read_csv(doe_path)
    # exp_ids = doe_df["exp_id"].values.astype(int)
    exp_ids = [exp_id]

    X_list, Y_list, exp_list = [], [], []

    for exp_id in exp_ids:
        
        exp_folder = os.path.join(data_raw_path, f"Ref_{exp_id}")
        
        if not os.path.isdir(exp_folder):
            
            print(f"⚠️ Carpeta no encontrada: {exp_folder}")
            
            continue

        input_time = idxToSave[0]
        output_time = idxToSave[-1]

        try:

            op_in = pd.read_csv(os.path.join(exp_folder, f"Ref_{exp_id}_op_{input_time}.csv"), header=None, sep = sep).values
            temp_in = pd.read_csv(os.path.join(exp_folder, f"Ref_{exp_id}_temp_{input_time}.csv"), header=None, sep = sep).values
            input_img = np.stack([op_in, temp_in], axis=-1)  # (100, 100, 2)

            op_out = pd.read_csv(os.path.join(exp_folder, f"Ref_{exp_id}_op_{output_time}.csv"), header=None, sep = sep).values
            temp_out = pd.read_csv(os.path.join(exp_folder, f"Ref_{exp_id}_temp_{output_time}.csv"), header=None, sep = sep).values
            output_img = np.stack([op_out, temp_out], axis=-1)  # (100, 100, 2)

            X_list.append(input_img)
            Y_list.append(output_img)
            exp_list.append(exp_id)

        except FileNotFoundError as e:
            print(f"❌ Faltan archivos para Ref_{exp_id}: {e}")

    X = np.array(X_list)
    Y = np.array(Y_list)
    exp_id_arr = np.array(exp_list)

    print(f"\n✅ Total cargado: {len(X)} experimentos.")
    print(f"   X shape: {X.shape}, Y shape: {Y.shape}")

    return X, Y, exp_id_arr
    