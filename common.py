########################################################################
# 导入 Python 库
########################################################################
# 默认库
import glob
import argparse
import sys
import os
import itertools
import re

# 额外库
import numpy as np
import librosa
import librosa.core
import librosa.feature
import yaml

########################################################################


########################################################################
# 设置标准输入输出和日志
########################################################################
"""
标准输出会被记录到 "baseline.log" 文件中。
"""
import logging

logging.basicConfig(level=logging.DEBUG, filename="fe_dgcan.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


########################################################################


########################################################################
# 版本号
########################################################################
__versions__ = "1.0.0"
########################################################################


########################################################################
# 命令行参数解析
########################################################################
def command_line_chk():
    parser = argparse.ArgumentParser(description='如果没有指定选项参数，程序将无法正常运行。')
    parser.add_argument('-v', '--version', action='store_true', help="显示程序版本")
    parser.add_argument('-d', '--dev', action='store_true', help="运行开发模式")
    parser.add_argument('-e', '--eval', action='store_true', help="运行评估模式")

    args = parser.parse_args()
    if args.version:
        print("===============================")
        print("DCASE 2022 task 2 baseline\nversion {}".format(__versions__))
        print("===============================\n")
    if args.dev:
        flag = True
    elif args.eval:
        flag = False
    else:
        flag = None
        print("incorrect argument")
        print("please set option argument '--dev' or '--eval'")
    return flag
########################################################################


########################################################################
# 读取参数文件 parameter.yaml
########################################################################
def yaml_load():
    with open("fe_dgcan.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

########################################################################


########################################################################
# 文件 I/O
########################################################################
# 读取 wav 文件
def file_load(wav_name, mono=False):
    """
    读取 .wav 文件。

    wav_name : str
        目标 .wav 文件路径
    mono : boolean
        当加载多声道文件且该参数为 True 时，返回的音频会被合成为单声道

    return : numpy.array( float )
        音频数据
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


########################################################################


########################################################################
# 特征提取
########################################################################
def file_to_vectors(file_name,
                    n_mels=64,
                    n_frames=5,
                    n_fft=1024,
                    hop_length=512,
                    power=2.0):
    """
    将音频文件转换为特征向量数组。

    file_name : str
        目标 .wav 文件路径

    return : numpy.array( numpy.array( float ) )
        特征向量数组
        * dataset.shape = (样本数量, 特征向量维度)
    """
    # 计算特征维度
    dims = n_mels * n_frames

    # 使用 librosa 生成梅尔频谱图
    y, sr = file_load(file_name, mono=True)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 转换为对数梅尔能量
    log_mel_spectrogram = 20.0 / power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))

    # 计算总特征向量数量
    n_vectors = len(log_mel_spectrogram[0, :]) - n_frames + 1

    # 跳过过短的音频片段
    if n_vectors < 1:
        return np.empty((0, dims))

    # 通过拼接多帧生成特征向量
    vectors = np.zeros((n_vectors, dims))
    for t in range(n_frames):
        vectors[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[:, t : t + n_vectors].T

    return vectors


########################################################################


########################################################################
# 根据模式选择数据目录
########################################################################
def select_dirs(param, mode):
    """
    param : dict
        从 parameter.yaml 读取的参数

    return :
        如果运行开发模式：
            dirs : list [ str ]
                返回开发数据集的目录列表
        如果运行评估模式：
            dirs : list [ str ]
                返回评估数据集的目录列表
    """
    if mode:
        logger.info("load_directory <- development")
        query = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
    else:
        logger.info("load_directory <- evaluation")
        query = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f)]
    return dirs


########################################################################


########################################################################
# 获取设备 section 名称
########################################################################
def get_section_names(target_dir,
                      dir_name,
                      ext="wav"):
    """
    target_dir : str
        基础目录路径
    dir_name : str
        子目录名称
    ext : str (默认="wav")
        音频文件扩展名

    return :
        section_names : list [ str ]
            从音频文件名中提取的 section 名称列表
    """
    # 匹配文件
    query = os.path.abspath("{target_dir}/{dir_name}/*.{ext}".format(target_dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(query))
    # 提取 section 名称
    section_names = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('section_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return section_names


########################################################################


########################################################################
# 获取音频文件列表
########################################################################
def file_list_generator(target_dir,
                        section_name,
                        dir_name,
                        mode,
                        prefix_normal="normal",
                        prefix_anomaly="anomaly",
                        ext="wav"):
    """
    target_dir : str
        基础目录路径
    section_name : str
        音频文件的 section 名称
    dir_name : str
        子目录名称
    prefix_normal : str (默认="normal")
        正常数据目录前缀
    prefix_anomaly : str (默认="anomaly")
        异常数据目录前缀
    ext : str (默认="wav")
        音频文件扩展名

    return :
        如果是开发模式：
            files : list [ str ]
                音频文件列表
            labels : list [ boolean ]
                标签列表
                * normal/anomaly = 0/1
        如果是评估模式：
            files : list [ str ]
                音频文件列表
    """
    logger.info("target_dir : {}".format(target_dir + "_" + section_name))

    # 开发模式
    if mode:
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                                     dir_name=dir_name,
                                                                                                     section_name=section_name,
                                                                                                     prefix_normal=prefix_normal,
                                                                                                     ext=ext))
        normal_files = sorted(glob.glob(query))
        normal_labels = np.zeros(len(normal_files))

        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                                     dir_name=dir_name,
                                                                                                     section_name=section_name,
                                                                                                     prefix_normal=prefix_anomaly,
                                                                                                     ext=ext))
        anomaly_files = sorted(glob.glob(query))
        anomaly_labels = np.ones(len(anomaly_files))

        files = np.concatenate((normal_files, anomaly_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
        
        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n========================================")

    # 评估模式
    else:
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*.{ext}".format(target_dir=target_dir,
                                                                                                     dir_name=dir_name,
                                                                                                     section_name=section_name,
                                                                                                     ext=ext))
        files = sorted(glob.glob(query))
        labels = None
        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n=========================================")

    return files, labels
########################################################################
