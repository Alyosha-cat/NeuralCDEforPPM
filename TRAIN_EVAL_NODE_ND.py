import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle 

from NODE.tensor_utils import prefix_alter, prefix_alter_test


def load_checkpoint(model, path_to_checkpoint, train_or_eval, lr, device='cpu'):
    """Loads already trained model into memory with the 
    learned weights, as well as the optimizer in its 
    state when saving the model.

    https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html 

    Parameters
    ----------
    model : instance of the CRTP Transformer
        Should just be initialized with the correct initialization 
        arguments, like you would have done in the beginning. 
    path_to_checkpoint : string
        Exact path where the checkpoint is stored on disk. 
    train_or_eval : str, {'train', 'eval'}
        Indicating whether you want to resume training ('train') with the 
        loaded model, or you want to evaluate it ('eval'). The layers of 
        the model will be returned in the appropriate mode. 
    lr : float 
        Learning rate of the optimizer last used for training. 
    
    Returns
    -------
    model : ...
        With trained weights loaded. 
    optimizer : ... 
        With correct optimizer state loaded. 
    final_epoch_trained : int 
        Number of final epoch that the model is trained for. 
        If you want to resume training with the loaded model, 
        start from start_epoch = final_epoch_trained + 1. 
    final_loss : ... 
        Last loss of last epoch. Don't think you need it for resuming 
        training. 
    """

    if (train_or_eval!= 'train') and (train_or_eval!= 'eval'):
        print("train_or_eval argument should be either 'train' or 'eval'.")
        return -1, -1, -1, -1

    checkpoint = torch.load(path_to_checkpoint)
    # Loading saved weights of the model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # Loading saved state of the optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Loading number of last epoch that the saved model was trained for. 
    final_epoch_trained = checkpoint['epoch:']
    # Last loss of the trained model. 
    final_loss = checkpoint['loss']

    if train_or_eval == 'train':
        model.train()
    else: 
        model.eval()
        
    return model, optimizer, final_epoch_trained, final_loss

def train_eval(log_name, 
               tss_index,
               hidden_dim,
               device='cpu'):
    """Training and automatically evaluating the NDA version of the 
    CRTP-LSTM benchmark model with the parameters used in the SuTraN 
    paper. 

    Parameters
    ----------
    log_name : str
        Name of the event log on which the model is trained. Should be 
        the same string as the one specified for the `log_name` parameter 
        of the `log_to_tensors()` function in the 
        `Preprocessing\from_log_to_tensors.py` module. 
    tss_index : int 
        Zero-based index at which the time since start (tss) prefix 
        event feature was stored in the original, fully data-aware 
        datasets. The time since previous event (tsp) prefix event 
        feature is stored at index `tss_index+1`. These two time related 
        features are the only numerical prefix event features retained 
        in the non data-aware benchmark models, and their indices are 
        therefore needed to retrieve these two features. 
    """

    def load_dict(path_name):
        with open(path_name, 'rb') as file:
            loaded_dict = pickle.load(file)
        return loaded_dict


    # -----------------
    temp_string = log_name + '_cardin_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_dict = load_dict(temp_path)
    num_activities = cardinality_dict['concept:name'] + 2
    print(num_activities)

    # cardinality list prefix categoricals 
    temp_string = log_name + '_cardin_list_prefix.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_list_prefix = load_dict(temp_path)

    temp_string = log_name + '_cardin_list_suffix.pkl'
    temp_path = os.path.join(log_name, temp_string)
    # cardinality list suffix categoricals
    cardinality_list_suffix = load_dict(temp_path)

    temp_string = log_name + '_num_cols_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    # To retrieve the number of numerical featrues in the prefix and suffix events respectively 
    num_cols_dict = load_dict(temp_path)

    temp_string = log_name + '_cat_cols_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cat_cols_dict = load_dict(temp_path)

    temp_string = log_name + '_train_means_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    train_means_dict = load_dict(temp_path)

    temp_string = log_name + '_train_std_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)

    train_std_dict = load_dict(temp_path)

    mean_std_ttne = [train_means_dict['timeLabel_df'][0], train_std_dict['timeLabel_df'][0]]
    mean_std_tsp = [train_means_dict['suffix_df'][1], train_std_dict['suffix_df'][1]]
    mean_std_tss = [train_means_dict['suffix_df'][0], train_std_dict['suffix_df'][0]]
    # mean_std_tss_pref = [train_means_dict['prefix_df'][5], train_std_dict['prefix_df'][5]]
    # mean_std_tsp_pref = [train_means_dict['prefix_df'][6], train_std_dict['prefix_df'][6]]
    mean_std_rrt = [train_means_dict['timeLabel_df'][1], train_std_dict['timeLabel_df'][1]]
    num_numericals_pref = len(num_cols_dict['prefix_df'])
    num_numericals_suf = len(num_cols_dict['suffix_df'])

    num_categoricals_pref, num_categoricals_suf = len(cat_cols_dict['prefix_df']), len(cat_cols_dict['suffix_df'])


    dropout = 0.2
    batch_size = 1024


    # specifying path results and callbacks 
    backup_path = os.path.join(log_name, f"NODE_NDA_results_hermit_{hidden_dim}")
    os.makedirs(backup_path, exist_ok=True)


    # Loading train, validation and test sets 
    # Training set 
    temp_path = os.path.join(log_name, 'train_tensordataset.pt')
    train_dataset = torch.load(temp_path)

    # Validation set
    temp_path = os.path.join(log_name, 'val_tensordataset.pt')
    val_dataset = torch.load(temp_path)

    # Test set 
    temp_path = os.path.join(log_name, 'test_tensordataset.pt')
    test_dataset = torch.load(temp_path)

    # For the prefix event token features: slice out only 
    # the activity label, as well as the two time 
    # features (time since start and time since previous)
    exclusive_bound = tss_index+2
    train_dataset = (train_dataset[num_categoricals_pref-1], ) + (train_dataset[num_categoricals_pref][:, :, tss_index:exclusive_bound],) + train_dataset[num_categoricals_pref+1:]

    test_dataset = (test_dataset[num_categoricals_pref-1], ) + (test_dataset[num_categoricals_pref][:, :, tss_index:exclusive_bound],) + test_dataset[num_categoricals_pref+1:]

    val_dataset = (val_dataset[num_categoricals_pref-1], ) + (val_dataset[num_categoricals_pref][:, :, tss_index:exclusive_bound],) + val_dataset[num_categoricals_pref+1:]

    # dataset: [batch, seq_len, feature_dim]

    train_dataset = prefix_alter(train_dataset, num_activities)
    test_dataset = prefix_alter(test_dataset, num_activities)
    val_dataset = prefix_alter(val_dataset, num_activities)

    input_channels = train_dataset[0].shape[-1]
    # Creating TensorDataset for the training set 
    train_dataset = TensorDataset(*train_dataset)

    # Initializing model 
    import random
    # Set a seed value
    seed_value = 24

    # Set Python random seed
    random.seed(seed_value)

    # Set NumPy random seed
    np.random.seed(seed_value)

    # Set PyTorch random seed
    torch.manual_seed(seed_value)

    # If you are using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
        # Additional settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Import Benchmark model 
    from NODE.NODE import NODE_no_context

    model = NODE_no_context(
        input_channels= input_channels,
        num_activities=num_activities, 
        hidden_dim=hidden_dim,
        dropout=0.2, 
    )



    # Assign to GPU 
    model.to(device)

    # Initializing optimizer and learning rate scheduler 
    decay_factor = 0.96
    lr = 0.002

    # Optimizer and scheduler used by benchmark
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_factor)

    
    # Training procedure 
    start_epoch = 0
    num_epochs = 500 
    num_classes = num_activities 
    batch_interval = 4096

    from NODE.train_procedure import train_model

    train_model(model, 
                optimizer=optimizer, 
                train_dataset=train_dataset, 
                val_dataset=val_dataset, 
                start_epoch=start_epoch, 
                num_epochs=num_epochs ,
                num_classes=num_activities, 
                batch_interval=batch_interval, 
                path_name=backup_path, 
                num_categoricals_pref=1, 
                mean_std_ttne=mean_std_ttne,
                mean_std_tsp=mean_std_tsp, 
                mean_std_tss=mean_std_tss, 
                mean_std_rrt=mean_std_rrt, 
                batch_size=batch_size, 
                lr_scheduler_present=True, 
                lr_scheduler=lr_scheduler,
                device=device,
                backup_path = backup_path)

    
    # Re-initializing new model after training to load best callback
    model = NODE_no_context(
        input_channels= input_channels,
        num_activities=num_activities, 
        hidden_dim=hidden_dim,
        dropout=0.2, 
    )

    # Assign to GPU 
    model.to(device)

    # Specifying path of csv in which the training and validation results 
    # of every epoch are stored. 
    final_results_path = os.path.join(backup_path, 'backup_results.csv')

    # Determining best epoch based on the validation 
    # scores for RRT and Activity Suffix prediction
    df = pd.read_csv(final_results_path)
    dl_col = 'Activity suffix: 1-DL (validation)'
    rrt_col = 'RRT - mintues MAE validation'
    df['rrt_rank_val'] = df[rrt_col].rank(method='min').astype(int)
    df['dl_rank_val'] = df[dl_col].rank(method='min', ascending=False).astype(int)
    df['summed_rank_val'] = df['rrt_rank_val'] + df['dl_rank_val']

    # Retrieving the row with the best general performance 
    row_with_lowest_loss = df.loc[df['summed_rank_val'].idxmin()]
    # Retrieve the value of the 'epoch' column for that row
    epoch_value = row_with_lowest_loss['epoch']

    # The models are stored with the string underneath
    best_epoch_string = 'model_epoch_{}.pt'.format(int(epoch_value))
    best_epoch_path = os.path.join(backup_path, best_epoch_string)

    # Loadd best model into memory again 
    model, _, _, _ = load_checkpoint(model, path_to_checkpoint=best_epoch_path, train_or_eval='eval', lr=0.002)
    model.to(device)
    model.eval()

    # Running final inference on test set 
    from NODE.inference_procedure import inference_loop

    # Initializing directory for final test set results 
    results_path = os.path.join(backup_path, "TEST_SET_RESULTS")
    os.makedirs(results_path, exist_ok=True)

    inf_results = inference_loop(model=model, 
        inference_dataset=val_dataset,
        remaining_runtime_head=True, 
        outcome_bool=False, 
        num_categoricals_pref=1, 
        mean_std_ttne=mean_std_ttne, 
        mean_std_tsp=mean_std_tsp, 
        mean_std_tss=mean_std_tss, 
        mean_std_rrt=mean_std_rrt, 
        results_path=results_path, 
        val_batch_size=4096,
        device=device)
    
    # Retrieving the different metrics 
    # TTNE MAE metrics
    # avg_MAE1_stand, avg_MAE1_seconds, avg_MAE1_minutes, avg_MAE2_stand, avg_MAE2_seconds, avg_MAE2_minutes = inf_results[:6]
    avg_MAE_ttne_stand, avg_MAE_ttne_minutes = inf_results[:2]
    # # Inference Cross Entropy Activity Suffix prediction
    # avg_inference_CE = inf_results[6]
    # Average Normalized Damerau-Levenshtein distance Activity Suffix 
    # prediction
    avg_dam_lev = inf_results[2]


    # Percentage of validation instances for which the END token was 
    # predicted too early. 
    perc_too_early = inf_results[3]
    # Percentage of validation instances for which the END token was 
    # predicted too late. 
    perc_too_late = inf_results[4]
    # Percentage of validation instances for which the END token was 
    # predicted at the right moment. 
    perc_correct = inf_results[5]
    # Mean absolute lenght difference between predicted and actual 
    # suffix. 
    mean_absolute_length_diff = inf_results[6]
    # Avg num events that END token was predicted too early, averaged 
    # over all instances for which END was predicted too early. 
    mean_too_early = inf_results[7]
    # Avg num events that END token was predicted too late, averaged 
    # over all instances for which END was predicted too late. 
    mean_too_late = inf_results[8]
    
    # MAE standardized RRT predictions 
    avg_MAE_stand_RRT = inf_results[9]
    # MAE RRT converted to minutes
    avg_MAE_minutes_RRT = inf_results[10]

    # Printing averaged results 
    print("Avg MAE TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_ttne_stand, avg_MAE_ttne_minutes))
    # print("Avg MAE Type 2 TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE2_stand, avg_MAE2_minutes))
    # print("Avg Cross Entropy acitivty suffix prediction validation set: {}".format(avg_inference_CE))
    print("Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev))
    print("Percentage of suffixes predicted to END: too early - {} ; right moment - {} ; too late - {}".format(perc_too_early, perc_correct, perc_too_late))
    print("Too early instances - avg amount of events too early: {}".format(mean_too_early))
    print("Too late instances - avg amount of events too late: {}".format(mean_too_late))
    print("Avg absolute amount of events predicted too early / too late: {}".format(mean_absolute_length_diff))

    print("Avg MAE RRT prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_stand_RRT, avg_MAE_minutes_RRT))

    # Retrieving and storing dictionary of the metrics averaged over all 
    # test set instances (prefix-suffix pairs)
    avg_results_dict = {"MAE TTNE minutes" : avg_MAE_ttne_minutes, 
                        "DL sim" : avg_dam_lev, 
                        "MAE RRT minutes" : avg_MAE_minutes_RRT}
    path_name_average_results = os.path.join(results_path, 'averaged_results.pkl')


    
    # Retrieving and storing the dictionaries with the 
    # averaged results per prefix and suffix length
    results_dict_pref = inf_results[-2]
    results_dict_suf = inf_results[-1]

    path_name_prefix = os.path.join(results_path, 'prefix_length_results_dict.pkl')
    path_name_suffix = os.path.join(results_path, 'suffix_length_results_dict.pkl')
    with open(path_name_prefix, 'wb') as file:
        pickle.dump(results_dict_pref, file)
    with open(path_name_suffix, 'wb') as file:
        pickle.dump(results_dict_suf, file)
    with open(path_name_average_results, 'wb') as file:
        pickle.dump(avg_results_dict, file)




def test(log_name, 
        tss_index,
        hidden_dim,
        missingrate,
        device='cpu'):
    """Training and automatically evaluating the NDA version of the 
    CRTP-LSTM benchmark model with the parameters used in the SuTraN 
    paper. 

    Parameters
    ----------
    log_name : str
        Name of the event log on which the model is trained. Should be 
        the same string as the one specified for the `log_name` parameter 
        of the `log_to_tensors()` function in the 
        `Preprocessing\from_log_to_tensors.py` module. 
    tss_index : int 
        Zero-based index at which the time since start (tss) prefix 
        event feature was stored in the original, fully data-aware 
        datasets. The time since previous event (tsp) prefix event 
        feature is stored at index `tss_index+1`. These two time related 
        features are the only numerical prefix event features retained 
        in the non data-aware benchmark models, and their indices are 
        therefore needed to retrieve these two features. 
    """

    def load_dict(path_name):
        with open(path_name, 'rb') as file:
            loaded_dict = pickle.load(file)
        return loaded_dict


    # -----------------
    temp_string = log_name + '_cardin_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_dict = load_dict(temp_path)
    num_activities = cardinality_dict['concept:name'] + 2

    # cardinality list prefix categoricals 
    temp_string = log_name + '_cardin_list_prefix.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_list_prefix = load_dict(temp_path)

    temp_string = log_name + '_cardin_list_suffix.pkl'
    temp_path = os.path.join(log_name, temp_string)
    # cardinality list suffix categoricals
    cardinality_list_suffix = load_dict(temp_path)

    temp_string = log_name + '_num_cols_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    # To retrieve the number of numerical featrues in the prefix and suffix events respectively 
    num_cols_dict = load_dict(temp_path)

    temp_string = log_name + '_cat_cols_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cat_cols_dict = load_dict(temp_path)

    temp_string = log_name + '_train_means_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    train_means_dict = load_dict(temp_path)

    temp_string = log_name + '_train_std_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)

    train_std_dict = load_dict(temp_path)

    mean_std_ttne = [train_means_dict['timeLabel_df'][0], train_std_dict['timeLabel_df'][0]]
    mean_std_tsp = [train_means_dict['suffix_df'][1], train_std_dict['suffix_df'][1]]
    mean_std_tss = [train_means_dict['suffix_df'][0], train_std_dict['suffix_df'][0]]
    # mean_std_tss_pref = [train_means_dict['prefix_df'][5], train_std_dict['prefix_df'][5]]
    # mean_std_tsp_pref = [train_means_dict['prefix_df'][6], train_std_dict['prefix_df'][6]]
    mean_std_rrt = [train_means_dict['timeLabel_df'][1], train_std_dict['timeLabel_df'][1]]
    num_numericals_pref = len(num_cols_dict['prefix_df'])
    num_numericals_suf = len(num_cols_dict['suffix_df'])

    num_categoricals_pref, num_categoricals_suf = len(cat_cols_dict['prefix_df']), len(cat_cols_dict['suffix_df'])

    # specifying path results and callbacks 
    backup_path = os.path.join(log_name, f"NODE_NDA_results_{hidden_dim}")
    os.makedirs(backup_path, exist_ok=True)


    # Loading train, validation and test sets 
    # Training set 
    temp_path = os.path.join(log_name, 'train_tensordataset.pt')
    train_dataset = torch.load(temp_path)

    # Validation set
    temp_path = os.path.join(log_name, 'val_tensordataset.pt')
    val_dataset = torch.load(temp_path)

    # Test set 
    temp_path = os.path.join(log_name, 'test_tensordataset.pt')
    test_dataset = torch.load(temp_path)

    # For the prefix event token features: slice out only 
    # the activity label, as well as the two time 
    # features (time since start and time since previous)
    exclusive_bound = tss_index+2
    train_dataset = (train_dataset[num_categoricals_pref-1], ) + (train_dataset[num_categoricals_pref][:, :, tss_index:exclusive_bound],) + train_dataset[num_categoricals_pref+1:]

    test_dataset = (test_dataset[num_categoricals_pref-1], ) + (test_dataset[num_categoricals_pref][:, :, tss_index:exclusive_bound],) + test_dataset[num_categoricals_pref+1:]

    val_dataset = (val_dataset[num_categoricals_pref-1], ) + (val_dataset[num_categoricals_pref][:, :, tss_index:exclusive_bound],) + val_dataset[num_categoricals_pref+1:]

    # dataset: [batch, seq_len, feature_dim]

    train_dataset = prefix_alter(train_dataset, num_activities)
    test_dataset = prefix_alter(test_dataset, num_activities)
    val_dataset = prefix_alter(val_dataset, num_activities)

    # need to add prefix_alter_test version to simulate MCR missingness
    test_dataset = prefix_alter_test(test_dataset, missingrate=missingrate)

    input_channels = train_dataset[0].shape[-1]
    # Creating TensorDataset for the training set 
    train_dataset = TensorDataset(*train_dataset)

    # Initializing model 
    import random
    # Set a seed value
    seed_value = 24

    # Set Python random seed
    random.seed(seed_value)

    # Set NumPy random seed
    np.random.seed(seed_value)

    # Set PyTorch random seed
    torch.manual_seed(seed_value)

    # If you are using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
        # Additional settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Import Benchmark model 
    from NODE.NODE import NODE_linear as NODE_no_context
    # Re-initializing new model after training to load best callback
    model = NODE_no_context(
        input_channels= input_channels,
        num_activities=num_activities, 
        hidden_dim=hidden_dim,
        dropout=0.2, 
    )

    # Assign to GPU 
    model.to(device)

    # Specifying path of csv in which the training and validation results 
    # of every epoch are stored. 
    final_results_path = os.path.join(backup_path, 'backup_results.csv')

    # Determining best epoch based on the validation 
    # scores for RRT and Activity Suffix prediction
    df = pd.read_csv(final_results_path)
    dl_col = 'Activity suffix: 1-DL (validation)'
    rrt_col = 'RRT - mintues MAE validation'
    df['rrt_rank_val'] = df[rrt_col].rank(method='min').astype(int)
    df['dl_rank_val'] = df[dl_col].rank(method='min', ascending=False).astype(int)
    df['summed_rank_val'] = df['rrt_rank_val'] + df['dl_rank_val']

    # Retrieving the row with the best general performance 
    row_with_lowest_loss = df.loc[df['summed_rank_val'].idxmin()]
    # Retrieve the value of the 'epoch' column for that row
    epoch_value = row_with_lowest_loss['epoch']

    # The models are stored with the string underneath
    best_epoch_string = 'model_epoch_{}.pt'.format(int(epoch_value))
    best_epoch_path = os.path.join(backup_path, best_epoch_string)

    # Loadd best model into memory again 
    model, _, _, _ = load_checkpoint(model, path_to_checkpoint=best_epoch_path, train_or_eval='eval', lr=0.002)
    model.to(device)
    model.eval()

    # Running final inference on test set 
    from NODE.inference_procedure import inference_loop

    # Initializing directory for final test set results 
    results_path = os.path.join(backup_path, f"TEST_SET_RESULTS_{missingrate}")
    os.makedirs(results_path, exist_ok=True)

    inf_results = inference_loop(model=model, 
        inference_dataset=test_dataset,
        remaining_runtime_head=True, 
        outcome_bool=False, 
        num_categoricals_pref=1, 
        mean_std_ttne=mean_std_ttne, 
        mean_std_tsp=mean_std_tsp, 
        mean_std_tss=mean_std_tss, 
        mean_std_rrt=mean_std_rrt, 
        results_path=results_path, 
        val_batch_size=4096,
        device=device)
    
    # Retrieving the different metrics 
    # TTNE MAE metrics
    # avg_MAE1_stand, avg_MAE1_seconds, avg_MAE1_minutes, avg_MAE2_stand, avg_MAE2_seconds, avg_MAE2_minutes = inf_results[:6]
    avg_MAE_ttne_stand, avg_MAE_ttne_minutes = inf_results[:2]
    # # Inference Cross Entropy Activity Suffix prediction
    # avg_inference_CE = inf_results[6]
    # Average Normalized Damerau-Levenshtein distance Activity Suffix 
    # prediction
    avg_dam_lev = inf_results[2]


    # Percentage of validation instances for which the END token was 
    # predicted too early. 
    perc_too_early = inf_results[3]
    # Percentage of validation instances for which the END token was 
    # predicted too late. 
    perc_too_late = inf_results[4]
    # Percentage of validation instances for which the END token was 
    # predicted at the right moment. 
    perc_correct = inf_results[5]
    # Mean absolute lenght difference between predicted and actual 
    # suffix. 
    mean_absolute_length_diff = inf_results[6]
    # Avg num events that END token was predicted too early, averaged 
    # over all instances for which END was predicted too early. 
    mean_too_early = inf_results[7]
    # Avg num events that END token was predicted too late, averaged 
    # over all instances for which END was predicted too late. 
    mean_too_late = inf_results[8]
    
    # MAE standardized RRT predictions 
    avg_MAE_stand_RRT = inf_results[9]
    # MAE RRT converted to minutes
    avg_MAE_minutes_RRT = inf_results[10]

    # Printing averaged results 
    print("Avg MAE TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_ttne_stand, avg_MAE_ttne_minutes))
    # print("Avg MAE Type 2 TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE2_stand, avg_MAE2_minutes))
    # print("Avg Cross Entropy acitivty suffix prediction validation set: {}".format(avg_inference_CE))
    print("Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev))
    print("Percentage of suffixes predicted to END: too early - {} ; right moment - {} ; too late - {}".format(perc_too_early, perc_correct, perc_too_late))
    print("Too early instances - avg amount of events too early: {}".format(mean_too_early))
    print("Too late instances - avg amount of events too late: {}".format(mean_too_late))
    print("Avg absolute amount of events predicted too early / too late: {}".format(mean_absolute_length_diff))

    print("Avg MAE RRT prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_stand_RRT, avg_MAE_minutes_RRT))

    # Retrieving and storing dictionary of the metrics averaged over all 
    # test set instances (prefix-suffix pairs)
    avg_results_dict = {"MAE TTNE minutes" : avg_MAE_ttne_minutes, 
                        "DL sim" : avg_dam_lev, 
                        "MAE RRT minutes" : avg_MAE_minutes_RRT}
    path_name_average_results = os.path.join(results_path, 'averaged_results.pkl')


    
    # Retrieving and storing the dictionaries with the 
    # averaged results per prefix and suffix length
    results_dict_pref = inf_results[-2]
    results_dict_suf = inf_results[-1]

    path_name_prefix = os.path.join(results_path, 'prefix_length_results_dict.pkl')
    path_name_suffix = os.path.join(results_path, 'suffix_length_results_dict.pkl')
    with open(path_name_prefix, 'wb') as file:
        pickle.dump(results_dict_pref, file)
    with open(path_name_suffix, 'wb') as file:
        pickle.dump(results_dict_suf, file)
    with open(path_name_average_results, 'wb') as file:
        pickle.dump(avg_results_dict, file)