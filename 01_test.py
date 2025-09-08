########################################################################
# 导入 Python 自带的库
########################################################################
import os
import csv
import sys
import gc
########################################################################


########################################################################
# 导入第三方依赖库
########################################################################
import numpy as np
import scipy.stats
# 进度条
from tqdm import tqdm
# 评估指标
from sklearn import metrics

try:
    # 旧版本 sklearn 里的 joblib
    from sklearn.externals import joblib
except:
    # 新版本直接用 joblib
    import joblib
# 项目自定义库
import common as com
import keras_model
########################################################################


from feature_enhancement import feature_enhancement_model
from dgcan import DGCAN
from train_process import vae_pretraining

########################################################################
# 加载 parameter.yaml 配置文件
########################################################################
param = com.yaml_load()


#######################################################################


########################################################################
# 保存结果到 CSV 文件
########################################################################
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


########################################################################


########################################################################
# 主程序入口：01_test.py
########################################################################
if __name__ == "__main__":
    # 检查运行模式
    # "development": mode == True
    # "evaluation" : mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # 创建结果保存目录
    os.makedirs(param["result_directory"], exist_ok=True)

    # 加载测试数据目录
    dirs = com.select_dirs(param=param, mode=mode)

    # 初始化保存 AUC / pAUC 等指标的表格
    csv_lines = []

    if mode:
        performance_over_all = []

    # 遍历每个机器类别目录
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {target_dir}".format(target_dir=target_dir, idx=idx + 1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]

        print("============== 加载模型 ==============")
        # 加载训练好的模型
        model_file = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                machine_type=machine_type)
        if not os.path.exists(model_file):
            com.logger.error("{} 模型不存在".format(machine_type))
            sys.exit(-1)
        model = keras_model.load_model(model_file)
        model.summary()

        # 加载 section 名称（条件向量用）
        section_names_file_path = "{model}/section_names_{machine_type}.pkl".format(model=param["model_directory"],
                                                                                    machine_type=machine_type)
        trained_section_names = joblib.load(section_names_file_path)
        n_sections = trained_section_names.shape[0]

        # 加载异常分数的分布参数（Gamma 分布）
        score_distr_file_path = "{model}/score_distr_{machine_type}.pkl".format(model=param["model_directory"],
                                                                                machine_type=machine_type)
        shape_hat, loc_hat, scale_hat = joblib.load(score_distr_file_path)

        # 根据分布计算阈值（分位点法）
        decision_threshold = scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat,
                                                   scale=scale_hat)

        if mode:
            # 针对每个机器类别的结果
            csv_lines.append([machine_type])
            csv_lines.append(["", "AUC (source)", "AUC (target)", "pAUC",
                              "precision (source)", "precision (target)", "recall (source)", "recall (target)",
                              "F1 score (source)", "F1 score (target)"])
            performance = []

        dir_name = "test"

        # 获取测试集里的 section 名称（比如机号）
        section_names = com.get_section_names(target_dir, dir_name)

        for section_name in section_names:

            # 检查该 section 是否在训练时出现过
            # 如果没出现过，section_idx = -1
            temp_array = np.nonzero(trained_section_names == section_name)[0]
            if temp_array.shape[0] == 0:
                section_idx = -1
            else:
                section_idx = temp_array[0]

                # 加载测试文件和标签
            files, y_true = com.file_list_generator(target_dir=target_dir,
                                                    section_name=section_name,
                                                    dir_name=dir_name,
                                                    mode=mode)

            # 设置异常分数保存路径
            anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{section_name}_{dir_name}.csv".format(
                result=param["result_directory"],
                machine_type=machine_type,
                section_name=section_name,
                dir_name=dir_name)
            anomaly_score_list = []

            # 设置最终判别结果保存路径
            decision_result_csv = "{result}/decision_result_{machine_type}_{section_name}_{dir_name}.csv".format(
                result=param["result_directory"],
                machine_type=machine_type,
                section_name=section_name,
                dir_name=dir_name)
            decision_result_list = []

            if mode:
                # 用于区分 source / target 域
                domain_list = []

            print("\n============== 开始测试一个 section ==============")
            y_pred = [0. for k in files]
            for file_idx, file_path in tqdm(enumerate(files), total=len(files)):
                try:
                    # 提取特征
                    data = com.file_to_vectors(file_path,
                                               n_mels=param["feature"]["n_mels"],
                                               n_frames=param["feature"]["n_frames"],
                                               n_fft=param["feature"]["n_fft"],
                                               hop_length=param["feature"]["hop_length"],
                                               power=param["feature"]["power"])
                except:
                    com.logger.error("文件损坏: {}".format(file_path))

                # 构造 one-hot 条件向量
                condition = np.zeros((data.shape[0], n_sections), float)
                if section_idx != -1:
                    condition[:, section_idx: section_idx + 1] = 1

                # 把 1D 向量转成 2D“图片”
                data = data.reshape(data.shape[0], param["feature"]["n_frames"], param["feature"]["n_mels"], 1)

                # 模型预测
                p = model.predict(data)[:, section_idx: section_idx + 1]
                # 计算异常分数（基于 log 概率）
                y_pred[file_idx] = np.mean(np.log(np.maximum(1.0 - p, sys.float_info.epsilon))
                                           - np.log(np.maximum(p, sys.float_info.epsilon)))

                # 保存异常分数
                anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])

                # 保存判别结果（大于阈值判为异常）
                if y_pred[file_idx] > decision_threshold:
                    decision_result_list.append([os.path.basename(file_path), 1])
                else:
                    decision_result_list.append([os.path.basename(file_path), 0])

                if mode:
                    domain_list.append("source" if "source" in file_path else "target")

            # 输出异常分数结果
            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            com.logger.info("异常分数结果保存 ->  {}".format(anomaly_score_csv))

            # 输出判别结果
            save_csv(save_file_path=decision_result_csv, save_data=decision_result_list)
            com.logger.info("判别结果保存 ->  {}".format(decision_result_csv))

            if mode:
                # 提取 source 和 target 域的预测结果
                y_true_s = [y_true[idx] for idx in range(len(y_true)) if
                            domain_list[idx] == "source" or y_true[idx] == 1]
                y_pred_s = [y_pred[idx] for idx in range(len(y_true)) if
                            domain_list[idx] == "source" or y_true[idx] == 1]
                y_true_t = [y_true[idx] for idx in range(len(y_true)) if
                            domain_list[idx] == "target" or y_true[idx] == 1]
                y_pred_t = [y_pred[idx] for idx in range(len(y_true)) if
                            domain_list[idx] == "target" or y_true[idx] == 1]

                # 计算 AUC, pAUC, 精确率, 召回率, F1
                auc_s = metrics.roc_auc_score(y_true_s, y_pred_s)
                auc_t = metrics.roc_auc_score(y_true_t, y_pred_t)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
                tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(y_true_s, [1 if x > decision_threshold else 0 for x in
                                                                             y_pred_s]).ravel()
                tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(y_true_t, [1 if x > decision_threshold else 0 for x in
                                                                             y_pred_t]).ravel()
                prec_s = tp_s / np.maximum(tp_s + fp_s, sys.float_info.epsilon)
                prec_t = tp_t / np.maximum(tp_t + fp_t, sys.float_info.epsilon)
                recall_s = tp_s / np.maximum(tp_s + fn_s, sys.float_info.epsilon)
                recall_t = tp_t / np.maximum(tp_t + fn_t, sys.float_info.epsilon)
                f1_s = 2.0 * prec_s * recall_s / np.maximum(prec_s + recall_s, sys.float_info.epsilon)
                f1_t = 2.0 * prec_t * recall_t / np.maximum(prec_t + recall_t, sys.float_info.epsilon)

                # 保存该 section 的结果
                csv_lines.append([section_name.split("_", 1)[1],
                                  auc_s, auc_t, p_auc, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])

                performance.a
