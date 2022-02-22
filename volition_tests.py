import numpy as np
from time import time
import random
import logging
from ml.dev.volition.volition_fake_data import (
    generate_fake_scores,
    generate_fake_comparisons,
    generate_data_user,
)

from ml.dev.volition.volition_model import Volition
from ml.dev.volition.volition_losses import *
from ml.dev.volition.evaluation import plot_dist_results, plot_hist_results, plot_volition_results
import os


# ========== unit tests ===============
# ---------- volition_fake_data.py ----------------

def test_generate_fake_scores():
    p, v = generate_fake_scores(4, 3)
    assert isinstance(p, np.ndarray)
    print(p, v)
    assert len(p) == len(v)


def test_generate_fake_comparisons():
    nb_videos = 10
    vids_per_user = 4
    nb_users = 2

    distribution = [vids_per_user] * nb_users
    weights = [0, 0.1, 0.5, 1]
    all_comps_detail, all_comps_arr, comps_q, gt = generate_fake_comparisons(nb_videos, distribution, weights)
    print(gt)
    print(all_comps_detail)
    print(len(gt.get(1)[1]))
    print(len(all_comps_detail[1]))
    print(all_comps_arr)


def test_generate_data_user():
    nb_videos = 1000
    vids_per_user = 6
    nb_users = 3
    weights = [0, 0.1, 0.5, 1]

    user_comp_dics, user_ids, vid_vidx, comps_q, ground_truths = generate_data_user(nb_videos, nb_users, vids_per_user,
                                                                                    weights, "testspecial")

    print(len(user_comp_dics), len(user_ids), len(ground_truths))
    print(user_comp_dics)
    print("_______________________")
    print(user_ids)
    print("_______________________")
    print(vid_vidx)
    print("_______________________")
    print(comps_q)
    print("_______________________")

    print(ground_truths)

# ---------- volition_model.py ----------------

def test_init_volition_model(nb_videos=None,
                             vids_per_user=None, nb_users=None,
                             weights_list=None, CRITERES=None, test_mode=True,
                             device="cpu", lr_gen=0.01,
                             lr_node=0.001, lambd=0.5,
                             opt_name="Adam",
                             path_folder=None,
                             vol_threshold=0.5
                             ):

    vol_model = Volition(nb_vids=nb_videos,
                         nb_user=nb_users,
                         nb_vid_user=vids_per_user,
                         criteria=CRITERES,
                         weights_list=weights_list,
                         test_mode=test_mode,
                         device=device,
                         opt_name=opt_name,
                         lr_gen=lr_gen,
                         lr_node=lr_node,
                         lambd=lambd,
                         path_folder=path_folder,
                         vol_threshold=vol_threshold
                         )

    vol_model.set_allnodes()
    return vol_model


def test_show_model_data():
    vol_model = test_init_volition_model()
    # show info of user nodes
    print(vol_model.user_data)
    print("_______________________________________________")
    print(vol_model.nb_comps)
    print("_______________________________________________")
    print(vol_model.vid_vidx)
    print("_______________________________________________")
    print(vol_model.users_ids)
    print("_______________________________________________")
    print(vol_model.ground_truth)
    print("_______________________________________________")
    print(vol_model.nodes.keys())

    id = 0
    # show data node of user 0
    print(vol_model.nodes.get(id).volition_gt)
    print(vol_model.nodes.get(id).preference_gt)
    print(vol_model.nodes.get(id).nb_comps)
    print(vol_model.nodes.get(id).vid_batch1)
    print(vol_model.nodes.get(id).vid_batch2)
    print(vol_model.nodes.get(id).rating)
    print(vol_model.nodes.get(id).weights)
    print(vol_model.nodes.get(id).y_data)
    print(vol_model.nodes.get(id).crit_index)
    print(vol_model.nodes.get(id).vids)
    print(vol_model.nodes.get(id).mask)
    print(vol_model.nodes.get(id).volition_model)
    print(vol_model.nodes.get(id).preference_model)
    print(vol_model.nodes.get(id).vids)
    print(vol_model.nodes.get(id).age)


# ---------- volition_losses.py ----------------


def test_get_fit_loss():
    vol_model = test_init_volition_model()
    loss = get_fit_loss(vol_model, critx=-1)
    print("THIS IS LOSS TEST RESULT:", loss)


def test_reg_loss():
    vol_model = test_init_volition_model()

    reg = reg_loss(vol_model.nodes, vol_model.lambd, vol_model.nb_criteria)

    print(reg)


def test_ml_run(nb_videos=None,
                vids_per_user=None, nb_users=None,
                weights_list=None, CRITERES=None, test_mode=True,
                device="cpu", lr_gen=0.01,
                lr_node=0.001, lambd=0.5,
                nb_epochs=600, opt_name="Adam",
                path_folder=None, vol_threshold=0.5
                ):

    ml_run_time = time()
    vol_model = test_init_volition_model(nb_videos=nb_videos,
                                         vids_per_user=vids_per_user, nb_users=nb_users,
                                         weights_list=weights_list, CRITERES=CRITERES, test_mode=test_mode,
                                         device=device,
                                         lr_gen=lr_gen,
                                         lr_node=lr_node,
                                         lambd=lambd,
                                         opt_name=opt_name,
                                         path_folder=path_folder,
                                         vol_threshold=vol_threshold)

    dist, train_hist, dist_file = vol_model.ml_run(nb_epochs=nb_epochs)
    try:
        open(f'plots/{dist_file}/{dist_file}.pickle')
    except IOError:
        print(f'plots/{dist_file}/{dist_file}.pickle', "file doesn't exist")

    logging.info(f'ml_run() total time : {time() - ml_run_time}')

    return dist, train_hist, dist_file#, vol_model.performance_model()


def seedall(s):
    """seeds all sources of randomness"""
    reproducible = s >= 0
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic = reproducible
    torch.backends.cudnn.benchmark = not reproducible
    print("\nSeeded all to", s)


def create_experiment_direc(parent_dir, opt_name, test_mode=True):

    if test_mode:
        path_file_result = f'{opt_name}_{int(time())}'
        # Path
        path = os.path.join(parent_dir, path_file_result)
        try:
            os.mkdir(path)
            print("Directory '% s' created" % path_file_result)
        except FileExistsError:
            print("Directory ", path_file_result, " already exists")

        logging.basicConfig(filename=f"{parent_dir}{path_file_result}/logs_{path_file_result}.log", level=logging.DEBUG)
        return path_file_result

    else:
        logging.basicConfig(filename=f"{parent_dir}logs.log", level=logging.DEBUG)


def main_test_mode():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


    nb_videos = 100000
    vids_per_user = 10
    nb_users = 100
    weights_list = [0, 0.1, 0.5, 1]
    CRITERES = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

    device = "cpu"
    lr_gen = 0.001
    lr_node = 0.2
    lambd = 0.5

    opt_name = "Adam"   #"Adam" or "rmsprop"
    nb_epochs = 100

    # creation 'plots' and result directories for visualization...
    # Parent Directory path
    parent_dir = "plots/"


    # test_generate_fake_scores()
    # test_generate_fake_comparisons()
    # test_generate_data_user()

    # test_init_volition_model()
    # test_get_fit_loss()

    # test_reg_loss()
    #seedall(9996465)

    nb_crit_step = 2

    for i in range(1):

        path_file_result = create_experiment_direc(parent_dir, opt_name)
        dist, train_hist, dist_file = test_ml_run(nb_videos=nb_videos,
                                                  vids_per_user=vids_per_user, nb_users=nb_users,
                                                  weights_list=weights_list, CRITERES=CRITERES, test_mode=True,
                                                  device=device, lr_gen=lr_gen,
                                                  lr_node=lr_node, lambd=lambd,
                                                  nb_epochs=nb_epochs, opt_name=opt_name,
                                                  path_folder=path_file_result, vol_threshold=0.5)

        plot_hist_results(train_hist, parent_dir, nb_epochs, file=dist_file, test_mode=True)
        plot_dist_results(dist, opt_name, file_download=dist_file )
    #plot_volition_results(vol_scores, nb_users, dist_file)
    # test_show_model_data()


def main_real_data():

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    device = "cpu"
    lr_gen = 0.00001
    lr_node = 0.9
    lambd = 0.5

    opt_name = "Adam"   #"Adam" or "rmsprop"
    nb_epochs = 100

    weights_list = [0, 0.5, 1, 1.5, 2]
    CRITERES = ["reliability", "importance", "engaging", "pedagogy", "layman_friendly", "diversity_inclusion",
                      "backfire_risk", "better_habits", "entertaining_relaxing"]

    parent_dir = "tournesol_datasets/"
    create_experiment_direc(parent_dir, opt_name, test_mode=False)

    _, train_hist, _ = test_ml_run(weights_list=weights_list, CRITERES=CRITERES, test_mode=False,
                                   device=device, lr_gen=lr_gen,
                                   lr_node=lr_node, lambd=lambd,
                                   nb_epochs=nb_epochs, opt_name=opt_name,
                                   vol_threshold=0.5)
    plot_hist_results(train_hist, parent_dir, nb_epochs, test_mode=False)

if __name__ == "__main__":

    #torch.autograd.set_detect_anomaly(True)
    #main_real_data()
    main_test_mode()
