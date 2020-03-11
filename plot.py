import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arguments import get_args
import os

if __name__ == "__main__":

    args = get_args()
    eval_file_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/eval_success_rates.npy'
    eval_file_1_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/eval_success_rates_buffer1epoch.npy'
    eval_file_2_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/eval_success_rates_bufferall.npy'

    if not os.path.isfile(eval_file_path):
        print("Result file do not exist!")
    else:
        data = np.load(eval_file_path)
        data1 = np.load(eval_file_1_path)
        data2 = np.load(eval_file_2_path)
        print(data)

        # data_epoch = []
        # for i in range(len(data1)):
        #     if i%args.n_cycles == 0:
        #         data_epoch.append(data1[i].copy())
        # data_epoch = np.array(data_epoch)
        # data_epoch = data_epoch[:30]
        # print(data_epoch)
        # print(data_epoch.shape)
        # print(data.shape)
        x = np.linspace(0, len(data), len(data))
        x1 = np.linspace(0, len(data1), len(data1))
        x2 = np.linspace(0, len(data2), len(data2))

        mpl.style.use('ggplot')
        fig = plt.figure(1)
        fig.patch.set_facecolor('white')
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Test Success Rate', fontsize=16)
        plt.title(args.env_name, fontsize=20)

        plt.plot(x2, data2, color='blue', linewidth=2, label='DDPG+HER,buffer all')
        plt.plot(x, data, color='red', linewidth=2, label='DDPG+HER, buffer 10 epoch')
        plt.plot(x1, data1, color='green', linewidth=2, label='DDPG+HER, buffer 1 epoch')
        plt.legend(loc='lower right')

        plt.show()
