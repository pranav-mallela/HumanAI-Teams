from dependencies import *
from combiner import *
import random

def load_CIFAR10H(model_name):
    """ Loads the CIFAR-10H predictions (human and model) and true labels.
    """
    dirname = PROJECT_ROOT

    data_path = os.path.join(dirname, f'dataset/{model_name}.csv')
    data = np.genfromtxt(data_path, delimiter=',')

    true_labels = data[:, 0]
    human_counts = data[:, 1:11]
    model_probs = data[:, 11:]

    true_labels = true_labels.astype(int)

    return human_counts, model_probs, true_labels

def simulate_humans(human_counts, y_true, accuracy_list = accuracies, seed=0):
    rng = np.random.default_rng(seed)
    human_labels = []

    assert len(human_counts) == len(y_true), "Size mismatch"

    i = -1

    for data_point in human_counts:
        i += 1
        labels = []
        for accuracy in accuracy_list:
            if (rng.random() < accuracy):
                labels.append(y_true[i])
            else:
                prob = data_point
                prob[y_true] = 0
                if (np.sum(prob) == 0):
                    prob = np.ones(prob.shape)
                    prob[y_true[i]] = 0
                prob /= np.sum(prob)
                labels.append(rng.choice(range(len(data_point)), p = prob))
                
        human_labels.append(labels)
    
    return np.array(human_labels)

def get_acc(y_pred, y_true):
    """ Computes the accuracy of predictions.
    If y_pred is 2D, it is assumed that it is a matrix of scores (e.g. probabilities) of shape (n_samples, n_classes)
    """
    if y_pred.ndim == 1:
        return np.mean(y_pred == y_true)
    print("Invalid Arguments")

def check_parity(mode, conf_num, test_size, model_name, policy_name):

    # constructing (2x2 conf mat for each class j) from (dim 10x10 conf mat for all)
    for j in range(10):
        tr_p = conf_mat[mode][conf_num][j][j]
        tr_n = 0
        fa_p = 0
        fa_n = 0
        for p in range(10):
            for q in range(10):
                if(p!=j and q!=j):
                    tr_n += conf_mat[mode][conf_num][p][q]
                elif p==j and p==q:
                    continue
                elif p==j:
                    fa_p += conf_mat[mode][conf_num][p][q]
                else:
                    fa_n += conf_mat[mode][conf_num][p][q]
        tot = tr_p + tr_n + fa_p + fa_n

        # using fairness metrics as defined on a 2x2 conf mat
        dem_from_conf_mat[mode][conf_num][j] = (tr_p + fa_p)/tot #predicted positive/total sum
        acc_from_conf_mat[mode][conf_num][j] = (tr_p + tr_n)/tot #sum of diagonal/total sum
        tpp_from_conf_mat[mode][conf_num][j] = (tr_p)/(tr_p + fa_n) #true positive/predicted positive
        fpp_from_conf_mat[mode][conf_num][j] = (fa_p)/(tr_p + fa_n) #false positive/predicted positive
        ppv_from_conf_mat[mode][conf_num][j] = (tr_p)/(tr_p + fa_p) #true positive/positive condition

    # writing to outf    
    titles[mode].append(str(mode) + " " + str(test_size) + " " + str(model_name) + " " + policy_name)
    outf.write(titles[mode][conf_num] + "\n")
    outf.write(str(tr_p) + " " +  str(fa_p) + " " + str(tr_n) + " " + str(fa_n) + "\n")
    outf.write(str(conf_mat[mode][conf_num]) + "\n\n")
    outf.write("Proportion of positive predictions: \n" + str(dem_from_conf_mat[mode][conf_num]) + "\n")
    outf.write("Accuracy per class: \n" + str(acc_from_conf_mat[mode][conf_num]) + "\n")
    outf.write("TP proportion per class: \n" + str(tpp_from_conf_mat[mode][conf_num]) + "\n")
    outf.write("FP proportion per class: \n" + str(fpp_from_conf_mat[mode][conf_num]) + "\n")
    outf.write("PPV per class: \n" + str(ppv_from_conf_mat[mode][conf_num]) + "\n\n")

def main():

    n_runs = 10
    test_sizes = [0.999, 0.99, 0.9, 0.0001]

    out_fpath = './output/'
    os.makedirs(out_fpath, exist_ok=True)
    model_names = ['cnn_data']
############################################################
    test_size_num = -1
############################################################
    for test_size in test_sizes:
        test_size_num += 1
        for model_name in tqdm(model_names, desc='Models', leave=True):
            # Specify output files
            output_file_acc = out_fpath + f'{model_name}_accuracy_{str(accuracies)}_{int((1-test_size)*10000)}'

            # Load data
            human_counts, model_probs, y_true = load_CIFAR10H(model_name) 

            # Generate human output from human counts through simulation
            y_h = simulate_humans(human_counts, y_true, accuracy_list=accuracies)

            POLICIES = [
                ('single_best_policy', single_best_policy, False),
                ('mode_policy', mode_policy, False),
                ('weighted_mode_policy', weighted_mode_policy, False),
                ('select_all_policy', select_all_policy, False),
                ('random', random_policy, False),
                ('lb_best_policy', lb_best_policy, True),
                ('pseudo_lb_best_policy_overloaded', pseudo_lb_best_policy_overloaded, False)
            ]

            acc_data = []
            for i in tqdm(range(n_runs), leave=False, desc='Runs'):
                seed = random.randint(1, 1000)
                # Train/test split
                y_h_tr, y_h_te, model_probs_tr, model_probs_te, y_true_tr, y_true_te = train_test_split(
                    y_h, model_probs, y_true, test_size=test_size, random_state=i * seed)
#############################################################

                # picking the label that has the highest model probability
                # in order to evaluate the model's performance before combining human labels
                y_cnn_model = np.zeros((10000), dtype=int)
                ind=-1
                for sublist in model_probs:
                    ind += 1
                    y_cnn_model[ind] = np.where(sublist==max(sublist))[0][0]

#############################################################
                # Test over entire dataset
                y_h_te = y_h
                model_probs_te = model_probs
                y_true_te = y_true

                acc_h = get_acc(y_h_te[:, 0], y_true_te) # considering the accuracy of the best human only
                acc_m = get_acc(np.argmax(model_probs_te, axis=1), y_true_te)

                _acc_data = [acc_h, acc_m]
                
                add_predictions("True Labels", y_true_te)

                combiner = MAPOracleCombiner()

                combiner.fit(model_probs_tr, y_h_tr, y_true_tr)
                
                policy_num = -1

                for policy_name, policy, use_true_labels in POLICIES:
                    
####################################################
                    # There are test_sizes*no_of_policies number of confusion matrices of dim 10x10
                    # Because each policy combines human input differently for each test_size
                    policy_num += 1
                    conf_num = test_size_num*7 + policy_num
####################################################

                    humans = policy(combiner, y_h_te, y_true_te if use_true_labels else None, np.argmax(model_probs_te, axis=1), NUM_HUMANS, model_probs_te.shape[1])
                    
                    y_comb_te = combiner.combine(model_probs_te, y_h_te, humans)

##################################################
                    no_images = 10000
                    for image in range(no_images):
                        c_i = y_comb_te[image] # label of combined model
                        y_i = y_true_te[image] # ground truth
                        m_i = y_cnn_model[image] # label of CNN model without human labels
                        conf_mat[0][conf_num][c_i][y_i] += 1 # incrementing combined model Conf Mat
                        conf_mat[1][conf_num][m_i][y_i] += 1 # incrementing CNN model Conf Mat

                    acc_comb = get_acc(y_comb_te, y_true_te)

                    # running on one of the test sizes is complete
                    if i == n_runs-1:
                        # mode = 0 => combined model; mode = 1 => only cnn model
                        check_parity(0, conf_num, test_size, model_name, policy_name)
                        check_parity(1, conf_num, test_size, model_name, policy_name)
#################################################################

                    _acc_data.append(acc_comb)

                acc_data += [_acc_data]

            header_acc = ['human', 'model'] + [policy_name for policy_name, _, _ in POLICIES]
            with open(f'{output_file_acc}_{i}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header_acc)
                writer.writerows(acc_data)

# open file pointer to txt file to print conf mat and fairness metrics
outf = open("outf.txt", "w")

# 7 policies * 4 test sizes = 28
conf_mat = np.zeros((2, 28, 10, 10), dtype=int)

"""Parity Measures:"""
acc_from_conf_mat = np.zeros((2, 28, 10)) # accuracy
dem_from_conf_mat = np.zeros((2, 28, 10)) # demographic parity
tpp_from_conf_mat = np.zeros((2, 28, 10)) # true positive parity
fpp_from_conf_mat = np.zeros((2, 28, 10)) # false positive parity
ppv_from_conf_mat = np.zeros((2, 28, 10)) # positive prediction value
titles = [[],[]]

main()