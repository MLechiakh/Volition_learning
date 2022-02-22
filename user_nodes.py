from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ConstantLR

"""
User node class used in "volition_model.py"

"""


class UserNode:
    def __init__(self, vol_gt, prefer_gt, nb_comps, vid1, vid2, ratings, weights, y_data, crit_index, vids,
                 vol_model, prefer_model, age, lr_node, opt, test_mode=True):
        if test_mode:
            self.volition_gt = vol_gt
            self.preference_gt = prefer_gt
            self.noise_gt = self.volition_gt - self.preference_gt

        self.nb_comps = nb_comps
        self.vid_batch1 = vid1
        self.vid_batch2 = vid2
        self.rating = ratings
        self.weights = weights
        self.y_data = y_data
        self.crit_index = crit_index
        self.vids = vids
        self.volition_model = vol_model
        self.preference_model = prefer_model
        # self.mean = mean
        # self.std = std
        self.age = age  # number of epochs the node has been trained
        self.opt = opt(
            [
                {"params": self.volition_model},
                {"params": self.preference_model}
            ], lr=lr_node)

        #self.scheduler_lr_node = StepLR(self.opt, step_size=10, gamma=0.1)
        self.scheduler_lr_node = ReduceLROnPlateau(self.opt, 'min', patience=5)
        #self.scheduler_lr_node = ConstantLR(self.opt, factor=0.5, total_iters=10)
