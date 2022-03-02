import numpy as np
import logging
import csv

from ml.dev.volition.volition_utils import (
    get_batch_r,
    get_batch_w,
    get_batch_y,
    rescale_rating,
    reverse_idxs,
    get_all_vids,
    one_hot_vids,
    get_crit_index
)


def read_from_dataset(path_folder):
    rating_columns = ["reliability", "importance", "engaging", "pedagogy", "layman_friendly", "diversity_inclusion",
                      "backfire_risk", "better_habits", "entertaining_relaxing"]
    weight_columns = ["reliability_weight", "importance_weight", "engaging_weight", "pedagogy_weight",
                      "layman_friendly_weight", "diversity_inclusion_weight",
                      "backfire_risk_weight", "better_habits_weight", "entertaining_relaxing_weight"]
    other_columns = ["id", "user__user__username", "video_1__video_id", "video_2__video_id", "duration_ms"]

    with open(f'{path_folder}.csv', mode='r', newline='') as csv_data:
        csv_reader = csv.DictReader(csv_data)
        users_arr = []
        all_users = []
        weights = []
        for row in csv_reader:
            all_users += [row[other_columns[1]]]
            if row["duration_ms"] != '0.0' and float(row["duration_ms"]) < 900000.0:
                ratings = []
                for i in range(len(weight_columns)):
                    if row[rating_columns[i]] == "":
                        ratings.append(-1.0)
                        weights += [0]
                    else:
                        ratings.append(float(row[rating_columns[i]]))
                        weights += [float(row[weight_columns[i]])]
                temp = [(rating_columns[i], float(ratings[i]), float(weights[i])) for i in range(len(ratings))]
                watching_infos = [row[other_columns[1]], row[other_columns[2]], row[other_columns[3]],
                                  float(row[other_columns[4]])]
                user_raw = watching_infos + temp
                users_arr += [user_raw]

        users_arr = sorted(users_arr, key=lambda x: x[0])

    # map string columns to integers
    #print("allll ", len(np.unique(np.array(all_users))))
    ids_users = np.unique(np.array(users_arr)[:, 0]).tolist()
    print("number of users: ", len(ids_users))
    ids_vids = np.unique(np.concatenate((np.array(users_arr)[:,1], np.array(users_arr)[:, 2]), axis=0)).tolist()
    print("number of videos: ", len(ids_vids))
    users_arr_encoded = []
    for user_data in users_arr:
        temp = [ids_users.index(user_data[0]), ids_vids.index(user_data[1]), ids_vids.index(user_data[2])]
        users_arr_encoded += [temp + user_data[3:]]

    user_comps_dict = {}
    users_dict = {}
    temp = []
    uid = users_arr_encoded[0][0]
    temp += [users_arr_encoded[0]]
    count = 1
    for i in range(1, len(users_arr_encoded)):
        uid_next = users_arr_encoded[i][0]
        if uid == uid_next:
            temp += [users_arr_encoded[i]]
            count += 1
        else:
            if count > 1:
                user_comps_dict.update({uid: count})
                users_dict.update({uid: temp})
            uid = uid_next
            temp = []
            temp += [users_arr_encoded[i]]
            count = 1
    user_comps_dict.update({uid: count})
    users_dict.update({uid: temp})

    return users_dict, user_comps_dict, rating_columns, len(users_dict.keys())


def _generate_data_user(path_dataset, path_folder):
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
    all_user_comps_detail, comps_queries, criteria_list, nb_users = read_from_dataset(path_dataset)
    nb_criteria = len(criteria_list)
    arr_train, arr_test, train_nb_comps_ = split_data_train_test(all_user_comps_detail, comps_queries)

    gene_create_train_test_arr_dataset(nb_criteria, arr_train, "train", path_folder)
    gene_create_train_test_arr_dataset(nb_criteria, arr_test, "test", path_folder)
    arr_train = shape_data(criteria_list, arr_train)
    # all_user_comps_array = shape_data(all_user_comps_detail)

    user_comp_dics, user_ids, vid_vidx = distribute_data(arr_train, nb_users)
    # create_dataset(ground_truths, all_user_comps_detail, path)
    logging.info(f" DATA GENERATION: user {id}: comparison data of + ground truths are generated")
    return user_comp_dics, user_ids, vid_vidx, train_nb_comps_


def split_data_train_test(arr, nb_comp, percen_test=0.2):
    import math
    arr_train = arr.copy()
    arr_test = {}
    comp = nb_comp.copy()

    for uid, data in zip(arr.keys(), arr.values()):
        nb_comps_ = len(arr.get(uid))
        num_test = math.ceil(nb_comps_ * percen_test)
        # index = num_test * NB_CRITERIAS
        index = nb_comps_ - num_test
        arr_test.update({uid: arr.get(uid)[index:]})
        arr_train.update({uid: arr.get(uid)[:index]})
        comp.update({uid: (nb_comp.get(uid) - num_test)})

    return arr_train, arr_test, comp


def gene_create_train_test_arr_dataset(nb_criteria, arr, name_split, path_folder):
    import csv

    header = ["user_ID", "video_1", "video_2", "y_data", "Criterion", "rating", "weight"]
    data = [[arr.get(i)[ii][0], arr.get(i)[ii][1], arr.get(i)[ii][2], arr.get(i)[ii][3],
             arr.get(i)[ii][4 + j][0], arr.get(i)[ii][4 + j][1], arr.get(i)[ii][4 + j][2]]
            for i in (arr.keys()) for ii in range(len(arr.get(i))) for j in range(nb_criteria)]
    # open the file in the write mode
    folder_name = 'tournesol_runs/'
    with open(f'{folder_name}{path_folder}/{name_split}.csv', 'w', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)


def shape_data(criteria, arr):
    """Shapes data for distribute_data()/distribute_data_from_save()
    l_ratings : list of not None ratings ([0,100]) for one criteria, all users
    Returns : one array with 4 columns : userID, vID1, vID2, rating ([-1,1])
    """
    l_data = []

    for uid, comps in zip(arr.keys(), arr.values()):
        list_crit = [
            rating[:4] + [criteria.index(rating[4 + i][0])] + [_rescale_rating(rating[4 + i][1])] + [_encode_weights(rating[4 + i][2], rating[4 + i][1])]
            for rating in comps for i in range(len(criteria))]
        l_data += list_crit
    return np.asarray(l_data)

def _rescale_rating(r):

    if r == -1.0:
        return r
    else:
        #return (r+50) / 100
        return r / 100

def _encode_weights(w, r):

    if r != -1:
        if w == 0.:
            return 1.
        elif w == 1.:
            return 1.5
        else:
            return w
    else:
        return 0
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
    print(nb_user, len(user_ids))
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


#if __name__ == "__main__":
    #user_comp_dics, user_ids, vid_vidx, train_nb_comps = generate_data_user("comparison_database")
