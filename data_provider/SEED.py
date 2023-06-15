import scipy.io as scio
import os
import numpy as np

eeg_save_path = r'D:\data\SEED\np_dataset\preprocessed_eeg'
eeg_load_path = r'D:\data\SEED\Preprocessed_EEG'


def build_preprocessed_eeg_dataset(load_path, save_path):
    labels = scio.loadmat(os.path.join(load_path, 'label.mat'))['label'][0]
    # 遍历以.mat结尾的文件
    file_cnt = 0
    for file in os.listdir(load_path):
        if file.endswith('.mat') and file != 'label.mat':
            file_cnt += 1
            print('当前已处理到{}，总进度{}/{}'.format(file, file_cnt, 45))
            cur_exp_data = []
            cur_exp_labels = []
            # 读取.mat文件
            data = scio.loadmat(os.path.join(load_path, file))
            experiment_name = file.split('.')[0]
            # 读取数据
            for key in data.keys():
                if 'eeg' not in key:
                    continue
                cur_trial_data = data[key]
                length = len(cur_trial_data[0])
                pos = 0
                while pos + 200 <= length:
                    cur_exp_data.append(np.asarray(cur_trial_data[:, pos:pos + 200]))
                    raw_label = labels[int(key.split('_')[-1][3:]) - 1]  # 截取片段对应的 label，-1, 0, 1
                    cur_exp_labels.append(raw_label + 1)
                    pos += 200
            cur_exp_data = np.array(cur_exp_data)
            cur_exp_labels = np.array(cur_exp_labels)
            np.save(os.path.join(save_path, experiment_name + '_data.npy'), cur_exp_data)
            np.save(os.path.join(save_path, experiment_name + '_labels.npy'), cur_exp_labels)


if __name__ == '__main__':
    build_preprocessed_eeg_dataset(eeg_load_path, eeg_save_path)