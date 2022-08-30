import torch
from ml.dev.volition.evaluation import plot_dist_results, plot_hist_results
from ml.dev.volition.volition_tests import test_generate_fake_comparisons, test_generate_data_user

"""
Losses used in "volition_model.py"
"""


def get_fit_loss(pref_model, vol_model, rating, weights, glob_mean,
                 y_data, glob_cov, nb_crit, critx=-1):
    """Fitting loss for one request of one user

    Args:

    Returns:
        (float scalar tensor): fitting loss.
    """

    assert len(rating) == len(weights) == nb_crit
    assert len(pref_model) == len(vol_model)

    if critx != -1:  # loss for only one criteria
        print("code for learning one crtietera of index: ", nb_crit)
        # FIXME here

    else:
        noise = pref_model - vol_model
        temp = torch.matmul((noise - glob_mean).T, torch.linalg.inv(glob_cov))
        bradly = bradely_node(pref_model, rating, weights, nb_crit)
        y_data = y_data / torch.linalg.vector_norm(noise)

        loss = 1 / 2 * torch.matmul(temp, (noise - glob_mean)) - torch.log(bradly) + y_data
        print("calculated loss: ", loss)

    return loss


def bradely_node(prefer_model, rating, weights, nb_crit):
    assert len(rating) == nb_crit == len(weights)

    t = weights.sum() * torch.matmul(prefer_model, rating)
    return torch.sigmoid(t)


def norm_noise(node, nb_crit):
    norm_values = []
    noise = node.preference_model - node.volition_model
    for i in range(len(noise) - nb_crit):
        l2_norm = torch.linalg.vector_norm(noise[i:i + nb_crit, ])
        norm_values += l2_norm
    return torch.tensor(norm_values)


def predict(input, tens, mask=None):
    """Predicts score according to a model

    Args:
        input (bool 2D tensor): one line is a one-hot encoded video index
        tens (float tensor): tensor = model
        mask (bool tensor): one element is bool for using this comparison

    Returns:
        (2D float tensor): score of the videos according to the model
    """
    if input.shape[1] == 0:  # if empty input
        return torch.zeros((1, 1))
    if mask is not None:
        return torch.where(mask, torch.matmul(input.float(), tens), torch.zeros(1))
    return torch.matmul(input.float(), tens)


def perfrmance_model():
    import csv

    with open(f'plots/Adam_1637717562/data_Adam_1637717562.csv', mode='r', newline='') as csv_data:
        csv_reader = csv.DictReader(csv_data)
        # for i in csv_reader:
        #     print(i)
        user_vol_dic = {}
        uid_vol_arr = []
        one_vol_list = []
        count = 0
        for row in csv_reader:
            uid = row["user_ID"]
            if count == 0:
                uid_old = uid
            if count % 4 == 0 and count != 0:
                uid_vol_arr += [one_vol_list]
                one_vol_list = []
            if uid == uid_old:
                one_vol_list += [int(row["volition"])]

            else:
                uid_vol_arr += one_vol_list
                user_vol_dic.update({int(uid_old): uid_vol_arr})
                uid_vol_arr = []
                one_vol_list = []
                one_vol_list += [row["volition"]]

            uid_old = uid
            count += 1

    return user_vol_dic


def mean_std(experiment_name, mean_gt, std_gt):
    import csv
    import numpy as np
    import pickle

    v = len(mean_gt)
    # recover data from test_file
    with open(f'plots/{experiment_name}/test_{experiment_name}.csv', mode='r', newline='') as csv_data:
        csv_reader = csv.DictReader(csv_data)
        mean = []
        std = []
        c = 0
        for row in csv_reader:
            if c < v:
                mean += [float(row["mean_noise_per_crit"])]
                std += [float(row["std_noise_per_crit"])]
                c += 1
            else:
                break

        mean = np.linalg.norm(np.array(mean) - np.array(mean_gt))
        std = np.linalg.norm(np.array(std) - np.array(std_gt))
        print(mean, std)

    with open(f'plots/{experiment_name}/{experiment_name}.pickle', 'rb') as handle:
        run_result_dict = pickle.load(handle)
        dist_vol_gt_pred = []

        for dist in run_result_dict.values():
            dist_vol_gt_pred += [dist[3]]

    print(sum(dist_vol_gt_pred) / len(dist_vol_gt_pred))


def read_from_dataset(path_folder):
    import csv
    import numpy as np
    rating_columns = ["reliability", "importance", "engaging", "pedagogy", "layman_friendly", "diversity_inclusion",
                      "backfire_risk", "better_habits", "entertaining_relaxing"]
    ratings = []
    weight_columns = ["reliability_weight", "importance_weight", "engaging_weight", "pedagogy_weight",
                      "layman_friendly_weight", "diversity_inclusion_weight",
                      "backfire_risk_weight", "better_habits_weight", "entertaining_relaxing_weight"]
    weights = []
    other_columns = ["id", "user__user__username", "video_1__video_id", "video_2__video_id", "duration_ms"]

    with open(f'{path_folder}.csv', mode='r', newline='') as csv_data:
        csv_reader = csv.DictReader(csv_data)
        users_arr = []
        for row in csv_reader:
            ratings = []
            for i in range(len(weight_columns)):
                if row[rating_columns[i]] == "":
                    ratings.append(0.0)
                else:
                    ratings.append(float(row[rating_columns[i]]))
            weights = [float(row[weight_columns[i]]) for i in range(len(weight_columns))]
            temp = [(rating_columns[i], ratings[i], weights[i]) for i in range(len(ratings))]
            watching_infos = [row[other_columns[1]], row[other_columns[2]], row[other_columns[3]],
                              float(row[other_columns[4]])]
            user_raw = watching_infos + temp
            users_arr += [user_raw]

        users_arr = sorted(users_arr, key=lambda x: x[0])

    user_comps_dict = {}
    users_dict = {}
    temp = []
    uid = users_arr[0][0]
    temp += [users_arr[0]]
    count = 1
    for i in range(1, len(users_arr)):
        uid_next = users_arr[i][0]
        if uid == uid_next:
            temp += [users_arr[i]]
            count += 1
        else:
            user_comps_dict.update({uid: count})
            users_dict.update({uid: temp})
            uid = uid_next
            temp = []
            temp += [users_arr[i]]
            count = 1

    user_comps_dict.update({uid: count})
    users_dict.update({uid: temp})

    print(sum(len(i) for i in users_dict.values()))
    print(len(users_arr))
    print(sum([i for i in user_comps_dict.values()]))


if __name__ == "__main__":
    #read_from_dataset("comparison_database")

    mean_vector = [-0.0005781164509244263, 0.0006221878575161099, 0.0022539354395121336, 1.936503394972533e-06,
                  0.00025755283422768116, 0.0007707903278060257, 0.00133400852791965, 0.0288272462785244,
                  0.14722806215286255]
    std_factor = [0.1875688135623932, 0.07390553504228592, 0.1843082159757614, 0.12553288042545319, 0.08145280927419662,
                 0.0325317457318306, 0.13096390664577484, 0.185774028301239, 0.28350725769996643]

    mean_std("Adam_1656974847", mean_vector, std_factor)
    # plot_dist_results(None,"Adam optimizer", file_upload = "Adam_1641986778")

    # test_generate_fake_comparisons()

    # test_generate_fake_dataset("")

    # x = perfrmance_model()

    # print(x)

    # test_generate_data_user()
    # import pandas as pd
    # import matplotlib.pyplot as plt
    #
    # object = pd.read_pickle(r'plots/rmsprop_1640980292/gt_test_rmsprop_1640980292.pickle')
    # print(object)
    # q = object.get(0)[1] / object.get(0)[0]
    #
    # import seaborn as sns
    #
    # sns.set_style("white")
    #
    # # Import data
    # x1 = q[:, 0]
    # x2 = q[:, 1]
    # x3 = q[:, 2]
    # x4 = q[:, 3]
    #
    # # Plot
    # kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})
    #
    # plt.figure(figsize=(10, 7), dpi=80)
    # sns.distplot(x1, color="dodgerblue", label="f1", **kwargs)
    # sns.distplot(x2, color="orange", label="f2", **kwargs)
    # sns.distplot(x3, color="deeppink", label="f3", **kwargs)
    # sns.distplot(x4, color="blue", label="f4", **kwargs)
    #
    # plt.legend()
    # plt.show()
