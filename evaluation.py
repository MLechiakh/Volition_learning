import numpy as np
from time import time
import random
import logging
from ml.dev.volition.volition_fake_data import (
    generate_fake_scores,
    generate_fake_comparisons,
    generate_data_user
)

from ml.dev.volition.volition_model import Volition
from ml.dev.volition.volition_losses import *
import matplotlib.pyplot as plt
import numpy as np
import pickle

DISTANCES = [
    "gt_vol_gt_pref",
    "mod_vol_gt_pref",
    "mod_vol_mod_pref",
    "mod_vol_gt_vol",
    "mod_pref_gt_pref"
]


def get_style():
    """gives different line styles for plots"""
    styles = [">", "o", "v", "<", "*"]
    for i in range(10000):
        yield styles[i % 4]


def get_color():
    """gives different line colors for plots"""
    colors = ["red", "green", "blue", "grey", "yellow"]
    for i in range(10000):
        yield colors[i % 5]


STYLES = get_style()  # generator for looping styles
COLORS = get_color()


def _title_save(title=None, path=None, suff=".png"):
    ''' Adds title and saves plot '''
    if title is not None:
        plt.title(title)
    if path is not None:
        plt.savefig(path + suff, bbox_inches='tight')
    plt.clf()


def _legendize(y, x="Epochs"):
    ''' Labels axis of plt plot '''
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')


def plot_dist_results(dist_arrays, name_opt, file_download=None, file_upload=None, new_fig_name=None):
    # Load data (deserialize)
    if file_upload is not None and dist_arrays is None:
        with open(f'plots/{file_upload}/{file_upload}.pickle', 'rb') as handle:
            run_result_dict = pickle.load(handle)

    else:
        run_result_dict = dist_arrays

    gt_dist, dist_pref_vol, model_dist, dist_vol_gt_pred, dist_pref_gt_pred = [], [], [], [], []

    for dist in run_result_dict.values():
        gt_dist += [dist[0]]
        dist_pref_vol += [dist[1]]
        model_dist += [dist[2]]
        dist_vol_gt_pred += [dist[3]]
        dist_pref_gt_pred += [dist[4]]


    dist_array_users = [gt_dist, dist_pref_vol, model_dist, dist_vol_gt_pred, dist_pref_gt_pred]

    users = range(1, len(run_result_dict.keys()) + 1)
    for i in range(len(dist_array_users)):
        style, color = next(STYLES), next(COLORS)
        plt.scatter(
            users, dist_array_users[i], label=DISTANCES[i], marker=style, s=5, color=color
        )

    _legendize("Distances", "usersIDs")
    title = f"Euclidian distances between model and Ground truth \n volition and preferences'scores using {name_opt}"

    if new_fig_name is not None:

        path = f'plots/{file_download}/dist_{new_fig_name}'
    else:
        path = f'plots/{file_download}/dist_{file_download}'
    _title_save(title, path)
    plt.close()

def plot_hist_results(train_hist, folder_name, epoch, file="test", test_mode=True):

    epochs = range(1, epoch + 1)
    fit, reg, loss, l_mean, std = train_hist.values()
    plt.plot(epochs, fit, label="fit", linestyle="-", color="r")
    plt.plot(epochs, reg, label="reg", linestyle='-', color='g')
    plt.plot(epochs, loss, label="loss", linestyle='-', color='y')
    _legendize("Loss", "epochs")
    if test_mode:
        _title_save("Loss = fit + reg", f'{folder_name}{file}/loss_{file}')
    else:
        _title_save("Loss = fit + reg", f'{folder_name}{file}/loss_{file}')

    plt.close()
    plt.plot(epochs, l_mean, label="noise_mean", linestyle='-', color='r')
    plt.plot(epochs, std, label="noise_std", linestyle='-.', color='black')

    _legendize("mean/std", "epochs")
    if test_mode:
        _title_save("mean (resp. std) of the model noise's 'mean vector' (resp. 'std matrix')", f'{folder_name}{file}/noise_{file}')
    else:
        _title_save("mean (resp. std) of the model noise's 'mean vector' (resp. 'std matrix')", f'{folder_name}{file}/noise_{file}')

    plt.close()

def plot_volition_results(s, user, file):

    users = range(1, user + 1)
    plt.bar(users, s)
    _legendize("Accuracy", "Users")
    _title_save(f"Accuracy of volition prediction, total accuracy: {round(sum(s)/len(s), 2)}", f'plots/{file}/volition_{file}')
    plt.close()


# def show_results(self, fit_loss, reg, total_loss, nb_epochs):
#     optimizer = "RMSProp"
#     l_dist_models = self.dist_models()[0]
#     for uid, dist in zip(l_dist_models.keys(), l_dist_models.values()):
#         loginf(f"USER {uid} ---> vol_pref_noise: {dist[0]}, volition error:{dist[1]},"
#                f"preference error: {dist[2]}")
#         print(f"USER {uid} ---> vol_pref_noise: {dist[0]}, volition error:{dist[1]},"
#               f"preference error: {dist[2]}")
#
#     loginf(f"__________________User local scores ______________________")
#
#     for uid, node in zip(self.nodes.keys(), self.nodes.values()):
#         loginf(
#             f"USER {uid}: volition_scores: \n {node.volition_model}\n preference scores: \n{node.preference_model}\n")
#         print(
#             f"USER {uid}: volition_scores: \n {node.volition_model}\n preference scores: \n{node.preference_model}\n")
#
#     loginf(f" ___________________Noise params ______________________")
#
#     for i, t in enumerate(self.noise_mean.tolist()):
#         loginf(f"noise mean index{i}:{t}")
#     loginf(f"noise_std: {self.noise_std.item()}")
#     loginf(f"Total loss = {total_loss}, fitting_loss = {fit_loss}, reg_term = {reg}")
#
#     self.plot_loss(np.arange(nb_epochs),
#                    [self.history["fit"], self.history["reg"], self.history["loss"]],
#                    ["fit", "reg", "loss"], ["r-", "b-", "g-"], optimizer)
#
#     print("____________________final statistics_____________________")
#     print(self.dist_models()[0])
#     print("_______", self.dist_models()[1])
