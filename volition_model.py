import torch
from torch.utils.tensorboard import SummaryWriter
from time import time
import numpy as np
import os
import matplotlib.pyplot as plt
from logging import info as loginf
from .user_nodes import UserNode
from .volition_losses import get_fit_loss, get_fit_loss_unidim_vectors, reg_loss, round_loss
from ml.dev.volition.volition_fake_data import generate_data_user
from ml.dev.volition.dataset_tournesol import _generate_data_user
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ConstantLR, ExponentialLR
import pickle


def get_volition_model(nb_comp, nb_crit, device="cpu"):
    #return torch.zeros(nb_crit, requires_grad=True, device=device)
    #return (-1 - 1) * torch.rand(1, nb_crit, requires_grad=True, device=device) + 1
    return torch.distributions.uniform.Uniform(-1, 1).sample([nb_crit]).requires_grad_().to(device)


def get_preference_model(nb_comp, nb_crit, device="cpu"):
    #return torch.zeros(nb_crit, requires_grad=True, device=device)
    #return (-1 - 1) * torch.rand(1, nb_crit, requires_grad=True, device=device) + 1
    return torch.distributions.uniform.Uniform(-1, 1).sample([nb_crit]).requires_grad_().to(device)


def get_glob_noise_mean(nb_crit, device="cpu"):
    return torch.zeros(nb_crit, requires_grad=True, device=device)


def get_glob_noise_cov(nb_crit, device="cpu"):
    return torch.ones(nb_crit, requires_grad=True, device=device)


class Volition:

    def __init__(
            self,
            nb_vids=None,
            nb_user=None,
            nb_vid_user=None,
            criteria=None,
            weights_list=None,
            test_mode=True,
            device="cpu",
            opt_name="Adam",
            lr_gen=None,
            lr_node=None,
            lambd=None,
            path_folder=None,
            vol_threshold=None
    ):
        """
        nb_vids (int): number of different videos rated by
                        at least one contributor for this criteria
        vid_vidx (dictionnary): dictionnary of {video ID: video index}
        crit (str): comparison criteria learnt
        device (str): device used (cpu/gpu)
        verb (float): verbosity level
        """
        # initiate configuration model parameters

        self.test_mode = test_mode
        self.nb_videos = nb_vids
        self.nb_vid_user = nb_vid_user
        self.nb_user = nb_user
        self.criteria = criteria  # list of criteria (str)
        self.nb_criteria = len(self.criteria)  # number of parameters of the model
        self.weights_list = weights_list

        self.device = device  # device used (cpu/gpu)
        self.opt = self.set_otimizer(opt_name)  # optimizer

        self.lr_node = lr_node  # local learning rate (volitional scores)
        self.lr_gen = lr_gen
        self.lambd = lambd  # regularisation strength

        self.vol_factor = vol_threshold

        self.path_folder = path_folder  # result directory/files identification name : optName_timestamp
        self.noise_mean = get_glob_noise_mean(self.nb_criteria, device=self.device)
        self.noise_std = get_glob_noise_cov(self.nb_criteria, device=self.device)
        # TensorBoard Writer Setup
        #self.writer = SummaryWriter(log_dir=f"runs/{self.lr_gen}, {self.opt.__name__}")

        self.opt_gen = self.opt([{"params": self.noise_mean},
                                 {"params": self.noise_std}], lr=self.lr_gen)
        self.scheduler_gen = ReduceLROnPlateau(self.opt_gen, 'min', patience=5)
        #self.scheduler_gen = ConstantLR(self.opt_gen, factor=0.01, total_iters=5)
        #self.scheduler_gen = ExponentialLR(self.opt_gen, 0.001, last_epoch=- 1, verbose=False)
        self.history = {
            "fit": [],
            "reg": [],  # metrics
            "loss": [],
            "l2_mean": [],
            "std": []
        }

        self.nodes = {}

        # Inits attributes required for test_mode
        if test_mode:
            self.vid_per_user = nb_vid_user
            self.user_data, self.users_ids, self.vid_vidx, self.nb_comps, self.ground_truth = generate_data_user(
                self.nb_videos, self.nb_user, self.nb_vid_user, self.weights_list, self.path_folder, self.criteria,
                threshold=self.vol_factor)
        else:
            self.tournesol_dataset = "comparison_database"
            self.user_data, self.users_ids, self.vid_vidx, self.nb_comps = _generate_data_user(self.tournesol_dataset, self.path_folder)


    def set_otimizer(self, name):

        if name == "SGD":
            opt = torch.optim.SGD
        elif name == "rmsprop":
            opt = torch.optim.RMSprop
        else:
            opt = torch.optim.Adam

        return opt

    def _get_default(self, nb_comp):
        """Returns: - (default noise mean, default noise std, default volitional, model, default age)"""
        models_tensors = (
            # get_glob_noise_mean(self.nb_criteria, self.device),
            # get_glob_noise_cov(self.device),
            get_volition_model(nb_comp, self.nb_criteria, self.device),
            get_preference_model(nb_comp, self.nb_criteria, self.device),
            0  # age: number of epochs the node has been trained
        )
        return models_tensors

    def set_allnodes(self):
        """create a node model for each user data

        data_dic (5-uplet): _ user_data(dictionnary): {userID: (vID1_batch, vID2_batch, rating_batch,
                            weights_batch, crit_index_array, y_batch, single_vIDs, mask)}
                            _ user_ids (array)
                            _ vid_vidx (dict) : vid:vidx
                            _ comps_queries (dict) : user_id : number of comparaison queries
                            _  ground truth (multi-dimensional array) [[[volitions], [preferences]] for all users]+
        users_ids (int array): users IDs
        """

        if self.test_mode:
            self.nodes = {
                id: UserNode(
                    self.ground_truth.get(id)[0], self.ground_truth.get(id)[1], self.nb_comps.get(id), *data,
                    *self._get_default(self.nb_comps.get(id)), self.lr_node, self.opt)

                for id, data in zip(self.user_data.keys(), self.user_data.values())
            }
            print(f"user nodes are created with a total number {len(self.nodes)}")
            loginf(f"user nodes are created with a total number {len(self.nodes)}")
            loginf(f"node lr schedulers instances are set")
        else:
            self.nodes = {
                id: UserNode(
                    None, None, self.nb_comps.get(id), *data,
                    *self._get_default(self.nb_comps.get(id)), self.lr_node, self.opt, test_mode=False)

                for id, data in zip(self.user_data.keys(), self.user_data.values())
            }
            print(f"user nodes are created with a total number {len(self.nodes)}")
            loginf(f"user nodes are created with a total number {len(self.nodes)}")
            loginf(f"node lr schedulers instances are set")

    def ml_run(
            self,
            nb_epochs=1,
            # resume=False,
            # save=True,
    ):

        loginf("\n------------------------\nSTARTING TRAINING\n--------------------------------")

        time_train = time()
        # training loop
        c_reg = max(self.weights_list) * self.nb_criteria

        for epoch in range(1, nb_epochs + 1):
            self._set_lr()
            loginf("-----------------epoch {}/{}-----------------".format(epoch, nb_epochs))
            print("-----------------epoch {}/{}------------------".format(epoch, nb_epochs))
            time_ep = time()

            # FIXME checking cov is stricly positive
            # ----------------     loss function  -------------------------
            fit_loss = get_fit_loss_unidim_vectors(self)
            #fit_loss = get_fit_loss(self)
            regu_loss = reg_loss(self, c_reg=c_reg)
            print("reg_loss= ", regu_loss, "fit_loss= ", fit_loss)
            loss = fit_loss + regu_loss
            total_loss = round_loss(loss)

            self._print_losses(total_loss, fit_loss, regu_loss)
            # Gradient descent
            loss.backward()
            # print("Loss grad: ", loss.grad)
            self._do_step()

            print(f"epoch {epoch}:: Loss: {total_loss} --> mean: {self.noise_mean} --> std: {self.noise_std}")
            # -----------for tensorboard------------
            # self.writer.add_scalar('mean1', self.noise_mean[0], global_step=epoch)
            # self.writer.add_scalar('mean2', self.noise_mean[1], global_step=epoch)
            # self.writer.add_scalar('mean3', self.noise_mean[2], global_step=epoch)
            # self.writer.add_scalar('mean4', self.noise_mean[3], global_step=epoch)
            #
            # self.writer.add_scalar('std', self.noise_std, global_step=epoch)
            # self.writer.add_scalar('Loss', loss, global_step=epoch)
            # ---------------------------------------
            self._zero_opt()  # resetting gradients
            self._scheduler_lr(loss)
            self._update_hist(fit_loss, regu_loss)
            self._old(1)  # aging all nodes of 1 epoch

            loginf(f"epoch: {epoch}--> epoch time :{round(time() - time_ep, 2)}")
            loginf(f"Total loss = {total_loss}, fitting_loss = {fit_loss}, reg_term = {regu_loss}")
            loginf(f"noise mean: {self.noise_mean.tolist()} noise_std: {self.noise_std.tolist()} \n")
            for id, node in self.nodes.items():
                loginf(f"{id}: vol_vector:{node.volition_model.detach().tolist()}, pref_vector:"
                       f"{node.preference_model.detach().tolist()}")
            loginf("\n-------------------------\nEND OF epoch\n----------------------------")

        #self.writer.close()  # for for tensorboard
        # ----------------- end of training -------------------------------
        loginf("\n-------------------------\nEND OF TRAINING\n----------------------------")
        training = round(time() - time_train, 2)
        loginf(f"training time :{training}")
        if self.test_mode:
            dist_results = self.dist_models()
            parent_dir = "plots/"  # path of directory of plots
            file = open(f'{parent_dir}{self.path_folder}/settings_{self.path_folder}.txt', "w")
            try:
                file.write(f"nb_users: {self.nb_user}\nnb_criteria: {self.nb_criteria}\nnb_videos: {self.nb_videos}\n"
                           f"nb_rated_videos_per_user: {self.nb_vid_user}\ndevice: {self.device}\n"
                           f"nb_compar_per_user: {self.nb_comps.get(0)}\n"
                           f"weighting_list: {self.weights_list}\n\n"
                           f"lr_gen: {self.lr_gen}\nlr_node: {self.lr_node}\n"
                           f"lambda: {self.lambd} \nnb_epochs: {nb_epochs}\ntraining_time: {training}\n\n"
                           f"convergence_loss_value: {self.history.get('loss')[-1]}\n\n"
                           f"mean_vector: {self.noise_mean.tolist()}\nstd_factor: {self.noise_std.tolist()}\n\n"
                           f"vol_comparison_threshold: {self.vol_factor}"
                           )
                file.close()
            except FileNotFoundError:
                print("File", f'setting_file.txt', " not found")

        else:
            parent_dir = "tournesol_runs/"
            file = open(f'{parent_dir}{self.path_folder}/settings_{self.path_folder}.txt', "w")
            try:
                file.write(
                           f"lr_gen: {self.lr_gen}\nlr_node: {self.lr_node}\n"
                           f"lambda: {self.lambd} \nnb_epochs: {nb_epochs}\ntraining_time: {training}\n\n"
                           f"convergence_loss_value: {self.history.get('loss')[-1]}\n\n"
                           f"mean_vector: {self.noise_mean.tolist()}\nstd_factor: {self.noise_std.tolist()}\n\n"
                           f"vol_comparison_threshold: {self.vol_factor}"
                           )
                file.close()
            except FileNotFoundError:
                print("File", f'setting_file.txt', " not found")

        self.save_models()
        if self.test_mode:
            with open(f"plots/{self.path_folder}/{self.path_folder}.pickle", 'wb') as handle:
                pickle.dump(dist_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return dist_results, self.history, self.path_folder
        else:
            return None, self.history, self.path_folder

    def _set_lr(self):
        """Sets learning rates of optimizers"""
        for node in self.nodes.values():
            node.opt.param_groups[0]["lr"] = self.lr_node  # node optimizer
            # node.opt.param_groups[2]["lr"] = self.lr_vol_model  # node optimizer
            # node.opt.param_groups[1]["lr"] = self.lr_vol_model  # node optimizer
            node.opt.param_groups[1]["lr"] = self.lr_node  # node optimizer

        self.opt_gen.param_groups[0]["lr"] = self.lr_gen
        self.opt_gen.param_groups[1]["lr"] = self.lr_gen

    def _scheduler_lr(self, loss):

        self.scheduler_gen.step(loss)
        for node in self.nodes.values():
            node.scheduler_lr_node.step(loss)

    def _zero_opt(self):
        """Sets gradients of all models"""
        for node in self.nodes.values():
            node.opt.zero_grad(set_to_none=True)  # node optimizer
        self.opt_gen.zero_grad(set_to_none=True)  # general optimizer

    def _do_step(self):
        """Makes step for appropriate optimizer(s)"""
        # if fit_step:  # updating local or global alternatively
        for node in self.nodes.values():
            node.opt.step()  # node optimizer
        # else:
        self.opt_gen.step()

    def _print_losses(self, loss, fit, reg):
        """Prints losses into log info"""
        fit, reg = round_loss(fit, 2), round_loss(reg, 2)

        loginf(
            f"total loss : {loss}\nfitting : {fit}, "
            f"regularisation : {reg}"
        )

    def _update_hist(self, fit, reg):
        """Updates history (at end of epoch)"""
        self.history["fit"].append(round_loss(fit))
        self.history["reg"].append(round_loss(reg))
        self.history["loss"].append(round_loss(fit + reg))

        # norm_mean = model_norm(self.noise_mean, pow=(2, 0.5))

        self.history["l2_mean"].append(self.noise_mean.mean().tolist())
        self.history["std"].append(self.noise_std.tolist())


    def _old(self, epochs):
        """Increments age of nodes (during training)"""
        for node in self.nodes.values():
            node.age += epochs

    def plot_loss(self, l_nb_epochs, l_losses, l_loss_names, l_color_styles, optimizer):

        print()
        assert len(l_losses[0]) == len(l_losses[1]) == len(l_losses[2])

        for i in range(len(l_loss_names)):
            plt.plot(l_nb_epochs, l_losses[i], l_color_styles[i], label=l_loss_names[i])

        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("loss minimization using " + optimizer)
        plt.legend()
        plt.show()

    def dist_models(self, p=2, q=0.5):

        l_nodes_dist = {}
        for uid, node in zip(self.nodes.keys(), self.nodes.values()):
            with torch.no_grad():
                model_dist = (((node.volition_model - node.preference_model) ** p).abs().sum() ** q)
                gt_dist = ((abs(node.volition_gt - node.preference_gt) ** p).sum() ** q)
                dist_vol_gt_pred = ((node.volition_model - node.volition_gt).detach() ** p).abs().sum() ** q
                dist_pref_gt_pred = ((node.preference_model - node.preference_gt) ** p).abs().sum() ** q
                dist_pref_vol = ((node.volition_model - node.preference_gt) ** p).abs().sum() ** q
                l_nodes_dist.update(
                    {uid: [round_loss(gt_dist, 3), round_loss(dist_pref_vol, 3), round_loss(model_dist, 3),
                           round_loss(dist_vol_gt_pred, 3),
                           round_loss(dist_pref_gt_pred, 3)]}
                )
        return l_nodes_dist

    def performance_model(self, vol_factor=0.1):

        import csv

        with open(f'plots/{self.path_folder}/train_{self.path_folder}.csv', mode='r', newline='') as csv_data:
            csv_reader = csv.DictReader(csv_data)

            user_vol_dic = {}
            uid_vol_arr = []
            one_vol_list = []
            count = 0
            for row in csv_reader:
                uid = row["user_ID"]
                if count == 0:
                    uid_old = uid
                if count % self.nb_criteria == 0 and count != 0:
                    uid_vol_arr += [one_vol_list]
                    one_vol_list = []
                if uid == uid_old:
                    one_vol_list += [int(row["volition"])]
                else:
                    uid_vol_arr += one_vol_list
                    user_vol_dic.update({int(uid_old): uid_vol_arr})
                    uid_vol_arr = []
                    one_vol_list = []
                    one_vol_list += [int(row["volition"])]
                    uid_old = uid

                count += 1
        uid_vol_arr += [one_vol_list]
        user_vol_dic.update({int(uid_old): uid_vol_arr})
        user_accuracy = []
        for uid, node in zip(self.nodes.keys(), self.nodes.values()):
            with torch.no_grad():
                #noise = (node.preference_model - node.volition_model).abs()
                ratio_vol = (node.preference_model - node.volition_model).abs()
                vol_scores = torch.where(ratio_vol <= vol_factor, 1, 0)
                accuracy = np.bitwise_xor(user_vol_dic.get(uid), vol_scores.tolist())
                s = accuracy.sum()
                s = 1 - s / accuracy.size
                user_accuracy += [s]
        user_accuracy.sort(reverse=True)

        return user_accuracy


    def save_models(self):
        """Saves age and global and local weights, detached (no gradients)"""
        loginf("Saving models")
        local_data = {
            id: (node.volition_model.detach().tolist(), node.preference_model.detach().tolist())  # + node.age in case of torch model saving
            for id, node in self.nodes.items()
        }
        saved_data = (
            self.noise_mean.detach(),
            self.noise_std.detach(),
            local_data,
        )
        if self.test_mode:
            parent_dir = "plots/"  # path of directory of plots
            file = open(f'{parent_dir}{self.path_folder}/results_{self.path_folder}.txt', "w")

        else:
            parent_dir = "tournesol_runs/"
            file = open(f'{parent_dir}{self.path_folder}/results_{self.path_folder}.txt', "w")

        try:
            file.write("mean_noise_vector: "+str(saved_data[0].tolist())+"\n")
            file.write("std_noise_factor: "+ str(saved_data[1].tolist())+"\n\n")
            file.write("____________  user_id : (model_vol,model_pref)__________\n\n")
            for key, val in zip(saved_data[2].keys(), saved_data[2].values()):
                file.write(str(int(key))+" : " + str(val)+"\n")
            file.close()
        except FileNotFoundError:
            print("File", f'result_file.txt', " not found")

        #torch.save(saved_data, fullpath)
        loginf(f"Models saved in results_{self.path_folder}.txt")
