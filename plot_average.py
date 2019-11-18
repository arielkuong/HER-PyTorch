import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arguments import get_args
import os

seed = [155, 831, 468, 531, 172]

if __name__ == "__main__":

    args = get_args()
    eval_file_path_1 = args.save_dir + args.env + '/seed_' + str(seed[0]) + '/eval_success_rates.npy'
    eval_file_path_2 = args.save_dir + args.env + '/seed_' + str(seed[1]) + '/eval_success_rates.npy'
    eval_file_path_3 = args.save_dir + args.env + '/seed_' + str(seed[2]) + '/eval_success_rates.npy'
    eval_file_path_4 = args.save_dir + args.env + '/seed_' + str(seed[3]) + '/eval_success_rates.npy'
    eval_file_path_5 = args.save_dir + args.env + '/seed_' + str(seed[4]) + '/eval_success_rates.npy'

    if not os.path.isfile(eval_file_path_1) \
    or not os.path.isfile(eval_file_path_2) \
    or not os.path.isfile(eval_file_path_3) \
    or not os.path.isfile(eval_file_path_4) \
    or not os.path.isfile(eval_file_path_5):
        print("Result file do not exist!")
    else:
        data_len = 140
        data1 = np.load(eval_file_path_1)[:data_len]
        data2 = np.load(eval_file_path_2)[:data_len]
        data3 = np.load(eval_file_path_3)[:data_len]
        data4 = np.load(eval_file_path_4)[:data_len]
        data5 = np.load(eval_file_path_5)[:data_len]

        x = np.linspace(0, data_len, data_len)
        data_comb = [data1, data2, data3, data4, data5]

        data_mean = np.mean(data_comb, axis=0)
        #print(data_mean)
        data_std = np.std(data_comb, axis=0)
        #print(data_std)
        data_low = data_mean - data_std
        data_high = data_mean + data_std

        save_data_path = args.save_dir + args.env + '/Average_result'

        if not os.path.exists(save_data_path):
            os.mkdir(save_data_path)

        np.save(save_data_path + '/data_mean.npy', data_mean)
        np.save(save_data_path + '/data_high.npy', data_high)
        np.save(save_data_path + '/data_low.npy', data_low)

        mpl.style.use('ggplot')
        fig = plt.figure(1)
        fig.patch.set_facecolor('white')
        plt.xlabel('Cycles(50 episodes/cycle)', fontsize=16)
        plt.ylabel('Test Success Rate', fontsize=16)
        plt.title(args.env, fontsize=20)

        plt.plot(x, data_mean, color='blue', linewidth=2, label='HER+HGG')
        plt.fill_between(x, data_low, data_high, color='blue', alpha=0.1)
        plt.legend(loc='lower right')

        plt.show()
