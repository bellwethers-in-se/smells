from __future__ import print_function, division
import os, sys
from glob import glob
from pdb import set_trace
from pandas import DataFrame, read_csv
import numpy as np
from tabulate import tabulate

def list_communities():
    for var in ["bell", "tca", "tnb"]:
        files = glob(os.path.abspath(os.path.join(".", "godclass", var, "*.csv")))
        yield files


def plot_stuff():
    pd_list = {}
    for vars in list_communities():
        for var in vars:
            pd_list.update({var.split("/")[-1].split(".")[0]: DataFrame(
                sorted(read_csv(var)[["Name", "G"]].values, key=lambda x: x[0], reverse=True))})

        N = len(pd_list.keys())  # Find number of elements
        stats = np.zeros((N, N))  # Create a 2-D Array to hold the stats
        keys = sorted(pd_list, reverse=True)  # Find data sets (Sort alphabetically, backwards)
        for idx, key in enumerate(keys):  # Populate 2-D array
            for i, val in enumerate(pd_list[key][1].values):
                if not i == idx:  # Ensure self values are set to zero
                    stats[i, idx] = val

        stats = DataFrame(stats, columns=keys, index=keys)
        stats["Mean"] = stats.mean(axis=1)
        stats["Std"] = stats.std(axis=1)
        print(tabulate(stats, showindex=True, headers=stats.columns, tablefmt="fancy_grid"))
        set_trace()


if __name__ == "__main__":
    plot_stuff()
