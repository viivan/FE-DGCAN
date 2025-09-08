########################################################################
# 导入 Python 自带的库
########################################################################
import os
import sys
import gc
########################################################################


########################################################################
# 导入第三方依赖库
########################################################################
import numpy as np
import scipy.stats
from tqdm import tqdm

try:
    # 尝试从 sklearn 加载 joblib（旧版本 sklearn 的写法）
    from sklearn.externals import joblib
except:
    # 新版本 sklearn 直接用 joblib
    import joblib
# 项目自定义工具库
import common as com
import keras_model
########################################################################


# 导入项目中自己写的模块
from feature_enhancement import feature_enhancement_model
from dgcan import DGCAN
from train_process import train

########################################################################
# 加载配置文件 parameter.yaml
########################################################################
param = com.yaml_load()


########################################################################


########################################################################
# 可视化工具类（用来绘制和保存训练曲线）
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(7, 5))
        # 设置子图之间的间隔
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        绘制训练和验证的 loss 曲线

        loss : list [ float ]
            训练过程的 loss 序列
        val_loss : list [ float ]
            验证过程的 loss 序列
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("模型损失曲线")
        ax.set_xlabel("训练轮次 (Epoch)")
        ax.set_ylabel("Loss")
        ax.legend(["训练集", "验证集"], loc="upper right")

    def save_figure(self, name):
        """
        保存图像到文件
        name : str
            保存的文件路径（png）
        """
        self.plt.savefig(name)


########################################################################


########################################################################
# 将文件列表转成训练数据
########################################################################
def file_list_to_data(file_list,
                      msg="calc...",
                      n_mels=64,
                      n_frames=5,
                      n_hop_frames=1,
                      n_fft=1024,
                      hop_length=512,
                      power=2.0):
    """
    把文件列表转化为特征向量矩阵
    内部会调用 file_to_vectors() 逐个文件提取特征，并拼接起来

    file_list : list [ str ]
        wav 文件路径列表
    msg : str
        tqdm 进度条的提示信息
    return : numpy.array
        拼接后的数据矩阵
        shape = (样本数量, 特征维度)
    """
    # 每个样本的维度 = n_mels * n_frames
    dims = n_mels * n_frames

    # 遍历文件，逐个提取特征
    for idx in tqdm(range(len(file_list)), desc=msg):
        vectors = com.file_to_vectors(file_list[idx],
                                      n_mels=n_mels,
                                      n_frames=n_frames,
                                      n_fft=n_fft,
                                      hop_length=hop_length,
                                      power=power)
        # 取步长，降低冗余
        vectors = vectors[::n_hop_frames, :]
        if idx == 0:
            # 初始化大数组
            data = np.zeros((len(file_list) * vectors.shape[0], dims), float)
        # 把当前文件的特征填充到大数组里
        data[vectors.shape[0] * idx: vectors.shape[0] * (idx + 1), :] = vectors

    return data


########################################################################


# 特征增强模块需要的频率分量数
n_freq_components = param['feature_enhancement']['n_freq_components']
# 初始化源域和目标域的训练数据（这里先建空数组）
x_source = np.empty((0, param["feature"]["n_frames"] * param["feature"]["n_mels"]), float)
x_target = np.empty((1, param["feature"]["n_frames"] * param["feature"]["n_mels"]), float)
# 输入张量形状 (帧数, 频率维度, 通道数=1)
input_shape = (param["feature"]["n_frames"], param["feature"]["n_mels"], 1)

########################################################################
# 主程序入口：00_train.py
########################################################################
if __name__ == "__main__":
    # 判断运行模式
    # "development" 开发模式: mode == True
    # "evaluation" 评估模式: mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # 创建输出目录（如果不存在就创建）
    os.makedirs(param["model_directory"], exist_ok=True)

    # 初始化可视化器
    visualizer = visualizer()

    # 加载训练数据目录
    dirs = com.select_dirs(param=param, mode=mode)

    # 初始化和训练模型（这里用特征增强 + DGCAN）
    feature_enhancement_model = feature_enhancement_model(input_shape, n_freq_components, hidden_units=512)
    dgcan_model = DGCAN(input_shape, num_classes=2, alpha=0.5)
    train(dgcan_model, x_source, x_target, epochs=200)

    # 遍历每个机器类型目录
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {target_dir}".format(target_dir=target_dir, idx=idx + 1, total=len(dirs)))

        # 模型保存路径
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                     machine_type=machine_type)

        if os.path.exists(model_file_path):
            com.logger.info("模型已存在，跳过训练")
            continue

        # 历史训练曲线图
        history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory"],
                                                                  machine_type=machine_type)
        # 存放 section 名称的文件
        section_names_file_path = "{model}/section_names_{machine_type}.pkl".format(model=param["model_directory"],
                                                                                    machine_type=machine_type)
        # 存放异常分数分布的文件
        score_distr_file_path = "{model}/score_distr_{machine_type}.pkl".format(model=param["model_directory"],
                                                                                machine_type=machine_type)

        # 提取 section 名称
        section_names = com.get_section_names(target_dir, dir_name="train")
        unique_section_names = np.unique(section_names)
        n_sections = unique_section_names.shape[0]

        # 保存 section 名称
        joblib.dump(unique_section_names, section_names_file_path)

        # 生成训练数据集
        print("============== DATASET_GENERATOR ==============")
        n_files_ea_section = []  # 记录每个 section 的文件数

        data = np.empty((0, param["feature"]["n_frames"] * param["feature"]["n_mels"]), float)

        for section_idx, section_name in enumerate(unique_section_names):
            # 获取该 section 的文件列表
            # 在训练阶段，所有 y_true = 0（正常音频）
            files, y_true = com.file_list_generator(target_dir=target_dir,
                                                    section_name=section_name,
                                                    dir_name="train",
                                                    mode=mode)

            n_files_ea_section.append(len(files))

            # 提取特征
            data_ea_section = file_list_to_data(files,
                                                msg="generate train_dataset",
                                                n_mels=param["feature"]["n_mels"],
                                                n_frames=param["feature"]["n_frames"],
                                                n_hop_frames=param["feature"]["n_hop_frames"],
                                                n_fft=param["feature"]["n_fft"],
                                                hop_length=param["feature"]["hop_length"],
                                                power=param["feature"]["power"])

            data = np.append(data, data_ea_section, axis=0)

        # 计算文件和向量数量
        n_all_files = sum(n_files_ea_section)
        n_vectors_ea_file = int(data.shape[0] / n_all_files)

        # 构造 one-hot 条件向量
        condition = np.zeros((data.shape[0], n_sections), float)
        start_idx = 0
        for section_idx in range(n_sections):
            n_vectors = n_vectors_ea_file * n_files_ea_section[section_idx]
            condition[start_idx: start_idx + n_vectors, section_idx: section_idx + 1] = 1
            start_idx += n_vectors

        # 把一维向量变成二维“图片”输入网络
        data = data.reshape(data.shape[0], param["feature"]["n_frames"], param["feature"]["n_mels"], 1)

        # 开始训练模型
        print("============== MODEL TRAINING ==============")
        model = keras_model.get_model(param["feature"]["n_frames"],
                                      param["feature"]["n_mels"],
                                      n_sections,
                                      param["fit"]["lr"])

        model.summary()

        history = model.fit(x=data,
                            y=condition,
                            epochs=param["fit"]["epochs"],
                            batch_size=param["fit"]["batch_size"],
                            shuffle=param["fit"]["shuffle"],
                            validation_split=param["fit"]["validation_split"],
                            verbose=param["fit"]["verbose"])

        # 用训练好的模型预测，得到异常分数分布
        y_pred = []
        start_idx = 0
        for section_idx in range(n_sections):
            for file_idx in range(n_files_ea_section[section_idx]):
                p = model.predict(data[start_idx: start_idx + n_vectors_ea_file, :, :, :])[:,
                    section_idx: section_idx + 1]
                # 用 log 概率来计算异常分数
                y_pred.append(np.mean(np.log(np.maximum(1.0 - p, sys.float_info.epsilon))
                                      - np.log(np.maximum(p, sys.float_info.epsilon))))
                start_idx += n_vectors_ea_file

        # 拟合异常分数的分布（Gamma 分布）
        shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(y_pred)
        gamma_params = [shape_hat, loc_hat, scale_hat]
        joblib.dump(gamma_params, score_distr_file_path)

        # 保存训练过程图像
        visualizer.loss_plot(history.history["loss"], history.history["val_loss"])
        visualizer.save_figure(history_img)

        # 保存模型
        model.save(model_file_path)
        com.logger.info("模型已保存 -> {}".format(model_file_path))
        print("============== END TRAINING ==============")

        # 释放内存
        del data
        del condition
        del model
        keras_model.clear_session()
        gc.collect()
