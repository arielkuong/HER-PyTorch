import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arguments import get_args
import os

if __name__ == "__main__":

    args = get_args()
    eval_file_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/eval_success_rates.npy'
    #eval_file_1_path = args.save_dir + args.env_name + '/eval_success_rates_old.npy'

    if not os.path.isfile(eval_file_path):
        print("Result file do not exist!")
    else:
        data = np.load(eval_file_path)
        #data1 = np.load(eval_file_1_path)
        print(data)

        data_epoch = []
        for i in range(len(data)):
            if i%args.n_cycles == 0:
                data_epoch.append(data[i].copy())
        data_epoch = np.array(data_epoch)
        print(data_epoch)
        print(data_epoch.shape)
        print(data.shape)
        x = np.linspace(0, len(data)/50, len(data)/50+1)
        #x1 = np.linspace(0, len(data1), len(data1))

        mpl.style.use('ggplot')
        fig = plt.figure(1)
        fig.patch.set_facecolor('white')
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Test Success Rate', fontsize=16)
        plt.title(args.env_name, fontsize=20)

        plt.plot(x, data_epoch, color='blue', linewidth=2, label='DDPG+HER')
        #plt.plot(x1, data1, color='red', linewidth=2, label='Pen-thin-long, from scratch')
        plt.legend(loc='lower right')

        plt.show()
