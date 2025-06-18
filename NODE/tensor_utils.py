import torch
import torch.nn as nn

def prefix_alter(dataset, num_activities):
    activity_emb_size = min(600, round(1.6 * (num_activities-2)**0.56))
    embed = nn.Embedding(num_embeddings=num_activities-1, embedding_dim=activity_emb_size, padding_idx=0)
    act_emb_pref = embed(dataset[0])
    num_ftrs_pref = dataset[1]
    padding = dataset[2]    # padding is at 3rd position as there is no categorical variables
    gtTTNE = dataset[-3]
    gtRM = dataset[-2]
    gtAct = dataset[-1]
    input = torch.cat((act_emb_pref, num_ftrs_pref), dim = -1).detach()
    return (input, padding, gtTTNE, gtRM, gtAct)

def prefix_alter_test(dataset, missingrate):
    if missingrate == 0:
        return dataset
    
    inputs = dataset[0]
    padding = dataset[1]
    key = (padding==False).sum(dim=1)

    # choose random idx to drop according to key
    drop = torch.zeros_like(inputs)

    for idx, count in enumerate(key):
        count = count.item()
        if count == 1:
            drop[idx] = inputs[idx]
            continue
        mask = torch.rand(count - 1) > missingrate
        num = (mask==False).sum()
        temp = torch.tensor([i for i in range(count-1)])[mask]
        drop[idx, 0:count-1-num] = inputs[idx][temp]
        drop[idx, count-1-num] = inputs[idx][count-1]

        padding[idx][count-num:] = True
    input = drop

    gtTTNE = dataset[-3]
    gtRM = dataset[-2]
    gtAct = dataset[-1]
    return (input, padding, gtTTNE, gtRM, gtAct)

def prefix_alter_test_others(dataset, missingrate):
    if missingrate == 0:
        return dataset
    act = dataset[0]
    num_ftrs_pref = dataset[1]
    padding = dataset[2]
    key = (padding==False).sum(dim=1)

    # choose random idx to drop according to key
    drop1 = torch.zeros_like(act)
    drop2 = torch.zeros_like(num_ftrs_pref)

    for idx, count in enumerate(key):
        count = count.item()
        if count == 1:
            drop1[idx] = act[idx]
            drop2[idx] = num_ftrs_pref[idx]
            continue
        mask = torch.rand(count - 1) > missingrate
        num = (mask==False).sum()
        temp = torch.tensor([i for i in range(count-1)])[mask]
        
        drop1[idx, 0:count-1-num] = act[idx][temp]
        drop2[idx, 0:count-1-num] = num_ftrs_pref[idx][temp]

        drop1[idx, count-1-num] = act[idx][count-1]
        drop2[idx, count-1-num] = num_ftrs_pref[idx][count-1]

        padding[idx][count-num:] = True
    return (drop1,)+(drop2,)+(padding,)+dataset[3:]
