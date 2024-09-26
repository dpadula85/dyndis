#!/usr/bin/env python

import os
import csv
import sys
import torch
import numpy as np
import pandas as pd
from datasets import DimersSP2NoH, GradDimersSP2NoH
from dig.threedgraph.method import SphereNet
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run

from torch_geometric.loader import DataLoader

def flatten(lst):
    '''
    Recursive function to flatten a nested list.

    Parameters
    ----------
    lst: list.
        Nested list to be flattened.

    Returns
    -------                                             
    flattened: list.
        Flattened list.
    '''

    flattened = sum( ([x] if not isinstance(x, list)
                     else flatten(x) for x in lst), [] )

    return flattened 


if __name__ == '__main__':

    dataset = DimersSP2NoH()
    split_idx = torch.load("splits_sp2.pt")
    test_dataset = dataset[split_idx['test']]

    dataset = GradDimersSP2NoH()
    test_dataset1 = dataset

    gpu = torch.device('cpu')

    # Define model, loss, and evaluation
    model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
                      hidden_channels=128, out_channels=1, int_emb_size=64,
                      basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8,
                      out_emb_channels=256, num_spherical=3, num_radial=6, envelope_exponent=5,
                      num_before_skip=1, num_after_skip=2, num_output_layers=3)                 

    model.load_state_dict(
            torch.load("trainsp2_chk.pt", map_location=gpu)["model_state_dict"]
        )
    model.eval()

    # Get predictions
    batch = DataLoader(test_dataset, batch_size=8)

    y = []
    yhat = []
    mols = []
    times = []
    didxs = []
    aidxs = []
    shifts = []
    for data in batch:
        y.append(data.y)
        mols.append(data.mol)
        times.append(data.t)
        didxs.append(data.didx)
        aidxs.append(data.aidx)
        shifts.append(data.shift)

        pred = model(data)
        yhat.append(pred.detach().numpy())

    y = np.concatenate(y).reshape(-1)
    yhat = np.concatenate(yhat).reshape(-1)
    mols = flatten(mols)
    times = np.concatenate(times).reshape(-1)
    didxs = np.concatenate(didxs).reshape(-1)
    aidxs = np.concatenate(aidxs).reshape(-1)
    shifts = flatten(shifts)
    df = pd.DataFrame({
                "Mol" : mols,
                "Time / ps": times,
                "DonorIdx" : didxs,
                "AccptIdx" : aidxs,
                "J_DFT / meV" : y,
                "J_ML / meV" : yhat
           })

    df.to_csv("results_sp2.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
