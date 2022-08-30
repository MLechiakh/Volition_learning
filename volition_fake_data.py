import numpy as np
import random
import logging
import scipy.stats as st

from ml.dev.volition.volition_utils import (
    get_batch_r,
    get_batch_w,
    get_batch_y,
    rescale_rating,
    _unscale_rating,
    reverse_idxs,
    get_all_vids,
    one_hot_vids,
    get_crit_index
)

# evaluation features
CRITERES = ["A", "B"]#["A", "B", "C", "D"] # ["A"]
# number of evaluation features
NB_CRITERIAS = len(CRITERES)

# factor to judge volitional preferences
vol_factor = 0.2


def set_global_variables(criteria, vol_threshold):
    global CRITERES
    global NB_CRITERIAS
    global vol_factor
    CRITERES = criteria
    NB_CRITERIAS = len(criteria)
    vol_factor = vol_threshold

def generate_fake_scores(nb_crit, scale=0.5):
    """Creates fake volition and preference scores for simulation

    nb_features (int): number of video representation features "generated"
    scale (float): variance of generated global scores

    Returns:
        (float array): fake preference scores
        (float array): fake volition scores
        (float array): noise (difference of scores)
        float : mean of noise
        float : standard deviation of noise
    """

    volition_scores = np.random.normal(scale=scale, size=nb_crit)
    #noise = np.random.normal(scale=2, size=nb_crit)
    preference = np.random.normal(scale=scale, size=nb_crit)

    return volition_scores, preference #np.asanyarray(noise + volition_scores)


def generate_fake_y(loc=60000, scale=600000):
    return st.expon.rvs(loc=loc, scale=scale)


def _get_rd_rate(scores):
    """Gives a random comparison score

        scores (list of float): local preferences of the comparaison between videos i and j
        compx (int) : index of comparaison  prefer scores in argument _scores_
    Returns:
        (float): random comparison score
    """
    # print("scooore", scores)
    preference_arg = list(scores)

    class MyPdf(st.rv_continuous):

        def _pdf(self, r, preference_arg):
            dens = 1 / (1 + np.exp(-np.matmul(np.array([preference_arg]), np.array([r]))))
            return dens

        def _argcheck(self, preference_arg):
            return True  # to admit also negative values

    my_cv = MyPdf(a=-1, b=1, name="my_pdf")
    return my_cv.rvs(preference_arg)


def generate_fake_comparisons(nb_vids, distribution, weight_list,
                              scale=0.1, dens=0.5):
    """

    nb_vids (int): number of videos
    distribution (int list): list of videos rated by the user
    user_node (list of float couple): (volition, preferences)
    weight_list (float) : choice  list for rating weights; e.g: skip=0, unsure:0.1, confident=1, very confident: 2
    ## 0 is the neutral value, it means that no weight is given by the user,
    Returns:
        (list of lists): list of all comparisons
                    [   contributor_id: int, video_id_1: int, video_id_2: int, y_data: float,
                        criteria_index: int, rating weight: float, weight: int  ]
    """
    all_comps_array = []
    all_comps_detail = []
    all_idxs = range(nb_vids)

    ground_truths = {}
    nb_comps_queries = {}
    for uid, nb_vids in enumerate(distribution):

        comps_count = 0
        comps_detail_user = []
        pick_all_idxs = random.sample(all_idxs, nb_vids)  # all videos rated by user
        volition_scores, preference_scores = [], []

        for vidx1, video1 in enumerate(pick_all_idxs):
            following_videos = range(vidx1 + 1, len(pick_all_idxs))
            nb_comp = int(dens * (len(pick_all_idxs) - vidx1))  # number of comparisons
            pick_idxs = random.sample(following_videos, nb_comp)

            for vidx2 in pick_idxs:
                volition, preference = generate_fake_scores(NB_CRITERIAS, scale)
                # preference = preference / 2
                r = _get_rd_rate(preference)  # get random list of ratings based on preference scores
                if NB_CRITERIAS == 1:
                    r = [r]
                if len(r) != len(preference):
                    raise NameError('length of preferences and ratings arrays doesnt match..')
                rate = _unscale_rating(r)  # rescale to [0, 100]
                temp = [(CRITERES[i], rate[i], random.choice(weight_list)) for i in range(NB_CRITERIAS)]
                y_data = generate_fake_y() #(scale=1/np.linalg.norm(preference - volition))
                comp_detail = [uid, pick_all_idxs[vidx1], pick_all_idxs[vidx2], y_data] + temp
                comps_detail_user.append(comp_detail)
                volition_scores += [volition]
                preference_scores += [preference]
                comps_count += 1
        ground_truths[uid] = (np.array(volition_scores), np.array(preference_scores))
        nb_comps_queries[uid] = comps_count
        # logging.info(f'ground truth for user {uid} is generated')

        all_comps_detail.append(comps_detail_user)
        all_comps_array.append(select_data_user(comps_detail_user))
    return all_comps_detail, all_comps_array, nb_comps_queries, ground_truths


def split_data_train_test(arr, gt, nb_comp, percen_test=0.2):
    import math
    arr_train = arr.copy()
    arr_test = []
    gt_train = gt.copy()
    gt_test = {}
    comp = nb_comp.copy()

    for uid, data in enumerate(arr):
        nb_comps_ = len(arr[uid])
        assert nb_comp.get(uid) == nb_comps_
        num_test = math.ceil(nb_comps_ * percen_test)
        # index = num_test * NB_CRITERIAS
        index = nb_comps_ - num_test
        index_gt = len(gt.get(uid)[0]) - num_test
        arr_test += [arr[uid][index:]]
        arr_train[uid] = arr[uid][:index]
        gt_test_vol = (gt.get(uid)[0][index_gt:], gt.get(uid)[1][index_gt:])
        gt_train_vol = (gt.get(uid)[0][:index_gt], gt.get(uid)[1][:index_gt])
        gt_train.update({uid: gt_train_vol})
        gt_test.update({uid: gt_test_vol})
        comp.update({uid: (nb_comp.get(uid) - num_test)})

    return arr_train, arr_test, gt_train, gt_test, comp


def select_data_user(comparison_data):
    """Extracts not None comparisons

    comparison_data (list of int, couples)
    crit: str, name of criteria

    Returns:
    - list of all ratings for this criteria
        ie list of [contributor_id: int, video_id_1: int, video_id_2: int,
                    criteria: str (crit), score: float, weight: float]
    """
    l_ratings = []
    for i in range(len(comparison_data)):
        for j in range(len(comparison_data[i]) - 4):

            if comparison_data[i][4 + j][1] is not None:
                l_ratings += [[comparison_data[i][0], comparison_data[i][1], comparison_data[i][2],
                               comparison_data[i][3], CRITERES.index(comparison_data[i][4 + j][0]),
                               comparison_data[i][4 + j][1], comparison_data[i][4 + j][2]]]

    if len(l_ratings) == 0:
        logging.warning(f"No comparison for this user ")
    return l_ratings


def shape_data(arr):
    """Shapes data for distribute_data()/distribute_data_from_save()

    l_ratings : list of not None ratings ([0,100]) for one criteria, all users

    Returns : one array with 4 columns : userID, vID1, vID2, rating ([-1,1])
    """
    l_clear = []
    for comps in arr:
        l_clear += [rating[:4] + [CRITERES.index(rating[4 + i][0])] + [rescale_rating(rating[4 + i][1])] + [rating[4 + i][2]]
              for rating in comps for i in range(NB_CRITERIAS)]

    return np.asarray(l_clear, dtype=float)


def distribute_data(arr, nb_user, device="cpu"):
    """Distributes data on nodes according to user IDs for one criteria
        Output is not compatible with previously stored models,
           ie starts from scratch

    arr (2D array): all ratings for all users for one criteria
                        (one line is [userID, vID1, vID2, y_data, crit_index, rating, weight])
    device (str): device to use (cpu/gpu)


    Returns:
    - dictionnary {userID: (vID1_batch, vID2_batch, rating_batch,
                            weights_batch, crit_index_array,
                             y_batch, single_vIDs, mask)}
    - array of user IDs
    - dictionnary of {vID: video idx}
    """
    logging.info("Preparing data from scratch")

    # checking array structure..
    user_ids, first_index = np.unique(arr[:, 0], return_index=True)

    if len(arr) == 0:
        raise NameError("users' array is empty")
    if nb_user != len(user_ids):
        raise NameError("number of users in array data is not correct")

    user_dic = {}
    vid_vidx = reverse_idxs(get_all_vids(arr))

    for id in range(len(first_index) - 1):
        batch1 = one_hot_vids(vid_vidx, arr[first_index[id]:first_index[id + 1], 1], device)
        batch2 = one_hot_vids(vid_vidx, arr[first_index[id]:first_index[id + 1], 2], device)

        user_dic[user_ids[id]] = (
            batch1,
            batch2,
            get_batch_r(arr[first_index[id]:first_index[id + 1], :], device),
            get_batch_w(arr[first_index[id]:first_index[id + 1], :], device),
            get_batch_y(arr[first_index[id]:first_index[id + 1], :], device),
            get_crit_index(arr[first_index[id]:first_index[id + 1], :]),
            get_all_vids(arr[first_index[id]:first_index[id + 1], :]),
        )

    # add last user

    batch1 = one_hot_vids(vid_vidx, arr[first_index[-1]:len(arr), 1], device)
    batch2 = one_hot_vids(vid_vidx, arr[first_index[-1]:len(arr), 2], device)

    user_dic[user_ids[len(first_index) - 1]] = (
        batch1,
        batch2,
        get_batch_r(arr[first_index[-1]:len(arr), :], device),
        get_batch_w(arr[first_index[-1]:len(arr), :], device),
        get_batch_y(arr[first_index[-1]:len(arr), :], device),
        get_crit_index(arr[first_index[-1]:len(arr), :]),
        get_all_vids(arr[first_index[-1]:len(arr), :]),
    )

    return user_dic, user_ids, vid_vidx


def generate_data_user(
        nb_videos, nb_users, vids_per_user, weights_values, path, criteria, threshold=0.5, scale=0.5, dens=0.5):
    """ Generates fake input data for testing

    nb_vids (int): number of videos
    nb_user (int): number of users
    vids_per_user (int): number of videos rated by each user
    scale (float): variance/std of global scores

    Returns:
         (5-uplet): _ user_data(dictionnary): {userID: (vID1_batch, vID2_batch, rating_batch,
                            weights_batch, crit_index_array, y_batch, single_vIDs, mask)}
                    _ user_ids (array)
                    _ vid_vidx (dict) : vid:vidx
                    _ comps_queries (dict) : user_id : number of comparaison queries
                    _ ground truth (multi-dimensional array) [[[volitions], [preferences]] for all users]+

    """
    set_global_variables(criteria, threshold)
    distribution = [vids_per_user] * nb_users
    all_user_comps_detail, all_user_comps_array, comps_queries, ground_truths = generate_fake_comparisons(
        nb_videos, distribution, weights_values, scale=scale, dens=dens
    )
    arr_train, arr_test, gt_train, gt_test, train_nb_comps = split_data_train_test(all_user_comps_detail, ground_truths,
                                                                                   comps_queries)

    gene_create_train_test_arr_dataset(gt_train, arr_train, "train", path)
    gene_create_train_test_arr_dataset(gt_test, arr_test, "test", path)
    creat_test_gt_data(gt_test, path)
    arr_train = shape_data(arr_train)
    #all_user_comps_array = shape_data(all_user_comps_detail)
    user_comp_dics, user_ids, vid_vidx = distribute_data(arr_train, nb_users)
    #create_dataset(ground_truths, all_user_comps_detail, path)
    logging.info(f" DATA GENERATION: user {id}: comparison data of + ground truths are generated")
    return user_comp_dics, user_ids, vid_vidx, train_nb_comps, gt_train


def creat_test_gt_data(gt_test, path):
    import pickle

    folder_name = 'datasets'
    with open(f"plots/{path}/gt_test_{path}.pickle", 'wb') as handle:
        pickle.dump(gt_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


def gene_create_train_test_arr_dataset(gt, arr, name_split, path):
    import csv

    l_vol_or_pref = []
    summary_l_vol_or_pref = []
    noise = []
    for uid, v in zip(gt.keys(), gt.values()):
        #vol_ratio = np.abs(v[1]) - np.abs(v[0])  # - 1 # preference , volition
        vol_ratio = np.array([[np.linalg.norm(v[1][i] - v[0][i]) / NB_CRITERIAS] for i in range(len(v[1]))])
        noise += [v[1][i] - v[0][i] for i in range(len(v[0]))]
        vol_or_pref = np.where(vol_ratio < vol_factor, 1, 0)
        l_vol_or_pref += [vol_or_pref]
        summary_l_vol_or_pref += [vol_or_pref.sum() / len(vol_or_pref)]

    noise = np.asarray(noise)
    mu = np.mean(noise, axis=0)
    st = np.std(noise, axis=0)
    l_vol_or_pref = np.array(l_vol_or_pref)
    header = ["user_ID", "video_1", "video_2", "y_data", "Criterion", "volition", "rating", "weight",
              "mean_noise_per_crit", "std_noise_per_crit"]
    data = [[arr[i][ii][0], arr[i][ii][1], arr[i][ii][2], arr[i][ii][3],
             arr[i][ii][4 + j][0], l_vol_or_pref[i][ii][0], arr[i][ii][4 + j][1], arr[i][ii][4 + j][2], mu[j], st[j]]
            for i in range(len(l_vol_or_pref)) for ii in range(len(arr[i])) for j in range(NB_CRITERIAS)]
    # open the file in the write mode
    folder_name = 'datasets'
    with open(f'plots/{path}/{name_split}_{path}.csv', 'w', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)



def create_train_test_arr_dataset(gt, arr, name_split, path):
    import csv

    l_vol_or_pref = []
    summary_l_vol_or_pref = []
    noise = []
    for uid, v in zip(gt.keys(), gt.values()):
        #vol_ratio = np.abs(v[1]) - np.abs(v[0])  # - 1 # preference , volition
        vol_ratio = np.array([[np.linalg.norm(v[1][i] - v[0][i]) / NB_CRITERIAS] for i in range(len(v[1]))])
        noise += [v[1][i] - v[0][i] for i in range(len(v[0]))]
        vol_or_pref = np.where(vol_ratio < vol_factor, 1, 0)
        l_vol_or_pref += [vol_or_pref]
        summary_l_vol_or_pref += [vol_or_pref.sum() / len(vol_or_pref)]

    noise = np.asarray([v[0] for v in noise])
    mu = np.mean(noise)
    st = np.std([noise])

    l_vol_or_pref = np.array(l_vol_or_pref)
    header = ["user_ID", "video_1", "video_2", "y_data", "Criterion", "volition", "rating", "weight", "mean_noise", "std_noise"]
    data = [[arr[i][ii][0], arr[i][ii][1], arr[i][ii][2], arr[i][ii][3],
             arr[i][ii][4 + j][0], l_vol_or_pref[i][ii][j], arr[i][ii][4 + j][1], arr[i][ii][4 + j][2], mu, st]
            for i in range(len(l_vol_or_pref)) for ii in range(len(arr[i])) for j in range(NB_CRITERIAS)]

    # open the file in the write mode
    folder_name = 'datasets'
    with open(f'plots/{path}/{name_split}_{path}.csv', 'w', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)

def create_dataset(gt, arr, path):
    import csv

    l_vol_or_pref = []
    summary_l_vol_or_pref = []
    for uid, v in zip(gt.keys(), gt.values()):
        vol_ratio = np.abs(v[1]) - np.abs(v[0])  # - 1 # preference , volition
        vol_or_pref = np.where(vol_ratio < vol_factor, 1, 0)
        l_vol_or_pref += [vol_or_pref]
        summary_l_vol_or_pref += [vol_or_pref.sum() / len(vol_or_pref)]

    l_vol_or_pref = np.array(l_vol_or_pref)
    header = ["user_ID", "video_1", "video_2", "y_data", "Criterion", "volition", "rating", "weight"]
    data = [[arr[i][ii][0], arr[i][ii][1], arr[i][ii][2], arr[i][ii][3],
             arr[i][ii][4 + j][0], l_vol_or_pref[i][ii][j], arr[i][ii][4 + j][1], arr[i][ii][4 + j][2]]
            for i in range(len(l_vol_or_pref)) for ii in range(len(arr[i])) for j in range(NB_CRITERIAS)]

    # open the file in the write mode
    with open(f'plots/{path}/data_{path}.csv', 'w', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)


if __name__ == "__main__":
    nb_videos = 10
    vids_per_user = 4
    nb_users = 2
    device = "cpu"
    scale = 1
    # node = {}
    # for i in range(nb_users):
    #     volition_param, preference_param,_,_,_ = generate_fake_scores(NB_CRITERIAS, 1)
    #     node[i] = [preference_param, volition_param]
    #
    # print(node)
    # distr = [vids_per_user] * nb_users
    # comps_detail, comps_array = generate_fake_comparisons(nb_videos, distr, node)
    # print(len(comps_detail))
    #
    # for i in comps_detail:
    #     print(i)
    #
    # print("______________________________________")
    # for j in comps_array:
    #     print(j)
    #
    # array = shape_data(comps_array[0])
    # a,b,c = distribute_data(array)
    # print(a)
    # print(b)
    # print(c)
    nodes = generate_data_user(nb_videos, nb_users, vids_per_user, scale)
