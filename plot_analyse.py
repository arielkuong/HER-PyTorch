import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arguments import get_args
import os

if __name__ == "__main__":

    args = get_args()
    file_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/record_training_actor_loss_epoch_avg.npy'
    file_1_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/record_training_critic_loss_epoch_avg.npy'
    # file_2_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/record_training_target_q_epoch_avg.npy'
    # file_3_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/record_training_real_q_epoch_avg.npy'
    file_4_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/record_training_samples_distance_ag_g_epoch_avg.npy'
    file_5_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/record_training_samples_distance_ag_g_epoch_std.npy'
    file_6_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/eval_success_rates_wo_norm.npy'

    actor_loss = np.load(file_path, allow_pickle=True)
    critic_loss = np.load(file_1_path, allow_pickle=True)
    # target_q = np.load(file_2_path, allow_pickle=True)
    # real_q = np.load(file_3_path, allow_pickle=True)
    sample_distance_avg = np.load(file_4_path, allow_pickle=True)
    sample_distance_std = np.load(file_5_path, allow_pickle=True)
    success_rate = np.load(file_6_path)
    # print(actor_loss.shape)
    # print(critic_loss.shape)
    # print(sample_obs.shape)
    # print(sample_action.shape)
    # print(sample_ag.shape)
    # print(sample_g.shape)
    # print(success_rate.shape)

    # Calculate the distance between achieved goal and goal
    sample_distance_avg = sample_distance_avg - sample_distance_avg.min()
    sample_distance_avg = sample_distance_avg / sample_distance_avg.max()
    sample_distance_std = sample_distance_std - sample_distance_std.min()
    sample_distance_std = sample_distance_std / sample_distance_std.max()

    x = np.linspace(0, len(actor_loss), len(actor_loss))
    x1 = np.linspace(0, len(critic_loss), len(critic_loss))
    # x2 = np.linspace(0, len(target_q), len(target_q))
    # x3 = np.linspace(0, len(real_q), len(real_q))
    x4 = np.linspace(0, len(sample_distance_avg), len(sample_distance_avg))
    x5 = np.linspace(0, len(sample_distance_std), len(sample_distance_std))
    x6 = np.linspace(0, len(success_rate), len(success_rate))


    mpl.style.use('ggplot')
    fig = plt.figure(1)
    fig.patch.set_facecolor('white')
    plt.xlabel('Epochs', fontsize=16)
    # plt.ylabel('Test Success Rate', fontsize=16)
    plt.title(args.env_name, fontsize=20)

    plt.plot(x, actor_loss/10, color='red', linewidth=2, label='actor loss/10')
    plt.plot(x1, critic_loss, color='blue', linewidth=2, label='critic loss')
    # plt.plot(x2, target_q, color='orange', linewidth=2, label='target q value')
    # plt.plot(x3, real_q, color='purple', linewidth=2, label='real q value')
    plt.plot(x4, sample_distance_avg, color='purple', linewidth=2, label='distance_ag_g_avg(norm)')
    plt.plot(x5, sample_distance_std, color='orange', linewidth=2, label='distance_ag_g_std(norm)')
    plt.plot(x6, success_rate, color='green', linewidth=2, label='success rate')
    plt.legend(loc='upper right')

    plt.show()
