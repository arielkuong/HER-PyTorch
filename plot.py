import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arguments import get_args
import os

if __name__ == "__main__":

    args = get_args()
    eval_file_path = args.save_dir + args.env_name + '/eval_success_rates.npy'
    eval_file_1_path = args.save_dir + args.env_name + '/eval_success_rates_old.npy'

    if not os.path.isfile(eval_file_path):
        print("Result file do not exist!")
    else:
        data = np.load(eval_file_path)
        data1 = np.load(eval_file_1_path)
        print(data)
        x = np.linspace(0, len(data), len(data))
        x1 = np.linspace(0, len(data1), len(data1))

        mpl.style.use('ggplot')
        fig = plt.figure(1)
        fig.patch.set_facecolor('white')
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Test Success Rate', fontsize=16)
        plt.title(args.env_name, fontsize=20)

        plt.plot(x, data, color='blue', linewidth=2, label='Pen, init with Egg')
        plt.plot(x1, data1, color='red', linewidth=2, label='Pen, from scratch')
        plt.legend(loc='lower right')

        plt.show()
