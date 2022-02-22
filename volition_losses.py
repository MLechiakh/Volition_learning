import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.exponential import Exponential

from logging import info as loginf

"""
Losses used in "volition_model.py"
"""



def get_fit_loss_unidim_vectors(vol_model, critx=-1):
    """Fitting loss for one node

    Args:

    Returns:
        (float scalar tensor): fitting loss.
    """
    fit_loss = 0
    if critx != -1:  # loss for only one criteria
        print("code for learning one crtietera of index: ", vol_model.nb_criteria)
        # FIXME here

    else:
        mvn = MultivariateNormal(vol_model.noise_mean, torch.diag(vol_model.noise_std))
        for uid, node in zip(vol_model.nodes.keys(), vol_model.nodes.values()):
            assert len(node.rating) == len(node.weights)
            assert len(node.preference_model) == len(node.volition_model)

            noise = (node.preference_model - node.volition_model)  # - vol_model.noise_mean) / vol_model.noise_std
            #noise = torch.nan_to_num(noise)
            #prob_noise = sum([1 / 2 * torch.matmul(torch.matmul((noise - vol_model.noise_mean), torch.linalg.inv(vol_model.noise_std * torch.eye(vol_model.nb_criteria))).T, (noise - vol_model.noise_mean))])  # for the case of vol_model and prefe_model unid vectors
            print("noise= ", noise)
            prob_noise = mvn.log_prob(noise)
            #prob_noise = pb_noise(noise, vol_model.noise_mean, vol_model.noise_std)
            ratings = [node.rating[i:i + vol_model.nb_criteria] for i in
                       range(0, len(node.rating), vol_model.nb_criteria)]
            weights = torch.nn.functional.normalize(
                torch.stack([node.weights[i:i + vol_model.nb_criteria] for i in
                             range(0, len(node.weights), vol_model.nb_criteria)]), dim=0)
            #print("weights= ", weights)
            bradly = bradly_node_unidim_vectors(node.volition_model, noise, ratings, weights, vol_model.nb_criteria)
            print("bradely_before= ", bradly)
            print("r ", len(ratings), " bradely ", len(bradly), " weighting ", len(weights), node.nb_comps)
            bradly = sum([torch.log(bradly[i]) for i in range(node.nb_comps)])
            print("bradely_after= ", bradly)
            noise_norm = torch.linalg.vector_norm(noise)  # for the case of vol_model and prefe_model unid vectors
            print("noise_norm= ", noise_norm)
            index = torch.tensor([i * vol_model.nb_criteria for i in range(node.nb_comps)])
            y_data = torch.index_select(node.y_data, 0, index)
            # print("y_data= ", y_data)
            # if len(y_data) > 1:
            #     normalized_y_data = (y_data - torch.min(y_data)) / (torch.max(y_data) - torch.min(y_data))
            # else:
            #     normalized_y_data = y_data / torch.abs(y_data)
            normalized_y_data = y_data / torch.sqrt(torch.sum(y_data ** 2))
            # normalized_y_data = y_data / 1000
            # print("y_data_normalized= ", normalized_y_data)
            #y_data_noise = normalized_y_data / noise_norm
            exp = Exponential(1 / noise_norm)
            y_data_noise = exp.log_prob(normalized_y_data)
            #print("y_data_noise_1= ", y_data_noise)
            for i in range(len(y_data_noise)):
                if y_data_noise[i] == float('inf') or y_data_noise[i] == -float('inf'):
                    with torch.no_grad():
                        y_data_noise[i] = normalized_y_data[i].item()

            y_data_noise = y_data_noise.sum()
            print("y_data_noise_2= ", y_data_noise)
            fit_loss += -prob_noise - bradly - y_data_noise
            #reg_loss += reg_loss_per_node(vol_model.nb_criteria, vol_model.lambd, uid, node, c_max=c_reg)
            print(f"user {uid}---> Loss: {fit_loss} y_data_noise: {y_data_noise}, "f"prob_noise: {prob_noise}, bradly: {bradly}")
            loginf(f"user {uid}--->fit_loss: {fit_loss}, y_data_noise: {y_data_noise},"
                   f" prob_noise: {prob_noise}, bradly: {bradly}")
            #print("reg, fit", reg_loss, fit_loss)
    return fit_loss


# to be used where dim(vol_model, pref_model) == (nb_comps, nb_crit)
def get_fit_loss(vol_model, critx=-1):
    """Fitting loss for one node

    Args:

    Returns:
        (float scalar tensor): fitting loss.
    """
    loss = 0
    if critx != -1:  # loss for only one criteria
        print("code for learning one crtietera of index: ", vol_model.nb_criteria)
        # FIXME here

    else:
        for uid, node in zip(vol_model.nodes.keys(), vol_model.nodes.values()):
            assert len(node.rating) == len(node.weights)
            assert len(node.preference_model) == len(node.volition_model)

            noise = ((node.preference_model - node.volition_model))  # - vol_model.noise_mean) / vol_model.noise_std
            loginf("_________________________ USER ", uid, " ___________________________")
            # print("noise: ", noise)
            # print("mean = ", vol_model.noise_mean, " std= ", vol_model.noise_std)
            prob_noise = sum([1 / 2 * torch.matmul(torch.matmul((n - vol_model.noise_mean), torch.linalg.inv(
                vol_model.noise_std * torch.eye(vol_model.nb_criteria))).T, (n - vol_model.noise_mean))
                              for n in noise])

            # prob_noise = sum([normal_dist.log_prob(n) for n in noise])
            # prob_noise = pb_noise(noise, vol_model.noise_mean, vol_model.noise_std)
            # print("prob_noise", prob_noise)
            # prob_noise = pb_noise(noise, vol_model.noise_mean, vol_model.noise_std, vol_model.nb_criteria)
            ratings = [node.rating[i:i + vol_model.nb_criteria] for i in
                       range(0, len(node.rating), vol_model.nb_criteria)]
            # print("ratings= ", ratings)
            weights = torch.nn.functional.normalize(
                torch.stack([node.weights[i:i + vol_model.nb_criteria].sum() for i in
                             range(0, len(node.weights), vol_model.nb_criteria)]), dim=0)
            # print("weights= ", weights)
            bradly = bradly_node(node.volition_model, noise, ratings, weights, node.nb_comps)
            bradly = sum([torch.log(bradly[i]) for i in range(node.nb_comps)])
            # print("bradly= ", bradly)
            noise_norm = torch.linalg.vector_norm(noise, dim=1)

            # print("noise_norm= ", noise_norm)
            index = torch.tensor([i * vol_model.nb_criteria for i in range(node.nb_comps)])
            y_data = torch.index_select(node.y_data, 0, index)
            y_data_noise = y_data / noise_norm
            # print("y_data_noise= ", y_data_noise)
            for i in range(len(y_data_noise)):
                if y_data_noise[i] == float('inf') or y_data_noise[i] == -float('inf'):
                    with torch.no_grad():
                        y_data_noise[i] = y_data[i].item()
            y_data_noise = y_data_noise.sum()
            loss += prob_noise - bradly + y_data_noise
            # print(f"user {uid}---> Loss: {loss} y_data_noise: {y_data_noise}, "
            # f"prob_noise: {prob_noise}, bradly: {bradly}")
            # loginf(f"user {uid}---> Loss: {loss} y_data_noise: {y_data_noise},"
            # f" prob_noise: {prob_noise}, bradly: {bradly}")

    return loss



def pb_noise(noise_tens, mean, cov):
    prob = []
    var = (cov ** 2)
    for n in noise_tens:
        prob += [((n - mean) ** 2) / (2 * var)]
    return sum(prob)


def reg_loss(vol_model, c_reg=1):
    reg_tens = []
    l2_norm_vol = 0
    for uid, node in zip(vol_model.nodes.keys(), vol_model.nodes.values()):
        weights = torch.tensor(
            [node.weights[i:i + vol_model.nb_criteria].sum() for i in
             range(0, len(node.weights), vol_model.nb_criteria)])
        t = ((2 * weights) / (c_reg * vol_model.nb_criteria + weights)).sum()  # / len(weights)
        l2_norm_vol += (node.volition_model ** 2).sum()
        reg_tens += [t * l2_norm_vol]

    reg = vol_model.lambd * sum(reg_tens)
    # print(f"regularization term for user {uid} is calculated : {reg}")
    loginf(f"user {uid} regularization component  is calculated : {reg}")

    return reg


def reg_loss_per_node(nb_criteria, lambd, uid, node, c_max=1):
    l2_norm_vol = 0
    weights = torch.tensor(
        [node.weights[i:i + nb_criteria].sum() for i in
         range(0, len(node.weights), nb_criteria)])
    t = ((2 * weights) / (c_max * nb_criteria + weights)).sum()  # / len(weights)
    l2_norm_vol += (node.volition_model ** 2).sum()
    reg_tens = [t * l2_norm_vol]

    reg = lambd * sum(reg_tens)
    # print(f"regularization term for user {uid} is calculated : {reg}")
    loginf(f"user {uid} regularization value  is calculated : {reg}")

    return reg


def get_sum_weight_node(node, nb_crit):
    weights_node = []
    if len(node.weights) % nb_crit != 0:
        raise NameError(f"length of weight tensor of user {id} is not multiple of number of criteria ")
    else:

        for i in range(0, len(node.weights), nb_crit):
            sum_weight = node.weights[i:i + nb_crit, ].sum()
            # alpha_k += (2 * sum_weight) / (C + sum_weight)

            weights_node += [sum_weight]

    return weights_node


def bradly_node(vol_model, noise, ratings, weights, nb_comp):
    assert len(ratings) == nb_comp == len(weights) == len(vol_model) == len(noise)
    t = [torch.matmul(vol_model[i] + noise[i], ratings[i]) for i in range(nb_comp)]
    t = [torch.sigmoid(weights[i] * t[i]) for i in range(nb_comp)]
    return t


def bradly_node_unidim_vectors(vol_model, noise, ratings, weights, nb_crit):
    assert nb_crit == len(vol_model) == len(noise)
    t = [torch.sigmoid(torch.matmul(vol_model + noise, torch.mul(weights[i], ratings[i]))) for i in range(len(ratings))]
    #t = [torch.sigmoid(weights[i] * t[i]) for i in range(len(ratings))]
    return t


def norm_noise(noise):
    norm_values = []
    for i in range(len(noise)):
        l2_norm = torch.linalg.vector_norm(noise)
        norm_values += l2_norm
    return torch.tensor(norm_values)


def round_loss(tens, dec=0):
    """from an input scalar tensor or int/float returns rounded int/float"""
    if type(tens) is int or type(tens) is float:
        return round(tens, dec)
    else:
        return round(tens.item(), dec)


def model_norm(model, pow=(2, 1)):
    """norm of a model (l2 squared by default)

    Args:
        model (float tensor): scoring model
        pow (float, float): (internal power, external power)
        vidx (int): video index if only one is computed (-1 for all)

    Returns:
        (float scalar tensor): norm of the model
    """
    p, q = pow
    return (model ** p).abs().sum() ** q
