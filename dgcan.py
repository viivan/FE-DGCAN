from keras.layers import Dense, Conv2D, Activation, GlobalAveragePooling2D,Multiply,BatchNormalization
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from feature_enhancement import feature_extractor,frequency_domain_enhancement

def sda_gcn(x, num_classes, attention_kernel_size=(3, 3)):
    """
    SDA-GCN模块：通过空间注意力机制提取与特定类别相关联的特征
    """
    # 使用2D卷积提取特征图
    feature_map = Conv2D(64, (3, 3), padding='same')(x)
    feature_map = BatchNormalization()(feature_map)
    feature_map = Activation('relu')(feature_map)

    # 使用Sigmoid激活函数得到类别特异性激活图
    attention_map = Conv2D(1, attention_kernel_size, padding='same', activation='sigmoid')(feature_map)

    # 特征图与激活图逐元素相乘
    feature_map = Multiply()([feature_map, attention_map])

    # 使用全局平均池化和1D卷积计算特征
    gap = GlobalAveragePooling2D()(feature_map)
    gap = Dense(128, activation='relu')(gap)

    # 输出类别预测
    output = Dense(num_classes, activation='softmax')(gap)

    return output

def dca_gcn(x, feature_map, num_classes, alpha=0.5):
    """
    DCA-GCN模块：通过自适应动态邻接矩阵进行细粒度的特征对齐
    """
    # 获取静态邻接矩阵
    static_adj_matrix = tf.matmul(x, x, transpose_b=True)

    # 动态卷积
    dynamic_adj_matrix = Conv2D(1, (3, 3), padding='same')(feature_map)
    dynamic_adj_matrix = tf.nn.sigmoid(dynamic_adj_matrix)

    # 结合静态邻接矩阵和动态邻接矩阵
    dynamic_adj_matrix = alpha * static_adj_matrix + (1 - alpha) * dynamic_adj_matrix

    # 融合邻接矩阵与特征
    feature_combined = tf.matmul(dynamic_adj_matrix, feature_map)

    # 输出
    output = Dense(num_classes, activation='softmax')(feature_combined)

    return output


def rann(x, y_true_coarse, y_true_fine, y_true_domain, num_classes, domain_classifier=True, alpha=0.5):
    """
    RANN模块：通过多任务学习和协方差对齐进行残差对齐
    """
    # 粗粒度分类器：分类源域与目标域
    coarse_classifier = Dense(num_classes, activation='softmax')(x)
    
    # 细粒度分类器：对更细化的偏移标签进行分类
    fine_classifier = Dense(num_classes, activation='softmax')(x)

    # 域分类器：判断样本来自源域还是目标域
    if domain_classifier:
        domain_classifier_output = Dense(1, activation='sigmoid')(x)

    # 计算协方差对齐损失
    # 假设源域特征为 x_s，目标域特征为 x_t
    x_s, x_t = x[:, :num_classes], x[:, num_classes:]
    covariance_loss = tf.reduce_mean(tf.square(tf.linalg.trace(tf.matmul(x_s, x_s, transpose_a=True) - tf.matmul(x_t, x_t, transpose_a=True))))

    # 计算分类损失
    # 使用交叉熵计算损失
    coarse_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true_coarse, coarse_classifier))
    fine_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true_fine, fine_classifier))

    # 如果使用域分类器，计算域分类损失
    if domain_classifier:
        domain_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true_domain, domain_classifier_output))
    else:
        domain_loss = 0.0

    # 总损失函数：粗粒度分类损失 + 细粒度分类损失 + 协方差对齐损失 + 域分类损失
    total_loss = coarse_loss + fine_loss + alpha * covariance_loss + domain_loss

    return total_loss


def DGCAN(input_shape, num_classes, alpha=0.5, execute_modules=1):
    """
    构建动态图卷积对抗网络（DGCAN）。
    """
    # 输入层
    x = Input(shape=input_shape)

    # 特征提取与增强部分
    if execute_modules != 1:
        feature_map = feature_extractor(input_shape)(x)
        enhanced_feature = frequency_domain_enhancement(feature_map)
    else:
        enhanced_feature = x
    
    # SDA-GCN (静态图卷积网络)
    if execute_modules != 1:
        sda_gcn_output = sda_gcn(enhanced_feature, num_classes)
    else:
        sda_gcn_output = None
    
    # DCA-GCN (动态跨域自适应图卷积网络)
    if execute_modules != 1:
        dca_gcn_output = dca_gcn(enhanced_feature, feature_map, num_classes, alpha=alpha)
    else:
        dca_gcn_output = None
    
    # RANN (残差对齐神经网络)
    if execute_modules != 1:
        rann_loss = rann(enhanced_feature, num_classes, domain_classifier=True, alpha=alpha)
    else:
        rann_loss = None

    # 构建模型
    if sda_gcn_output is not None and dca_gcn_output is not None and rann_loss is not None:
        model = Model(inputs=x, outputs=[sda_gcn_output, dca_gcn_output, rann_loss])
    elif sda_gcn_output is not None and dca_gcn_output is not None:
        model = Model(inputs=x, outputs=[sda_gcn_output, dca_gcn_output])
    elif sda_gcn_output is not None and rann_loss is not None:
        model = Model(inputs=x, outputs=[sda_gcn_output, rann_loss])
    elif dca_gcn_output is not None and rann_loss is not None:
        model = Model(inputs=x, outputs=[dca_gcn_output, rann_loss])
    elif sda_gcn_output is not None:
        model = Model(inputs=x, outputs=sda_gcn_output)
    elif dca_gcn_output is not None:
        model = Model(inputs=x, outputs=dca_gcn_output)
    elif rann_loss is not None:
        model = Model(inputs=x, outputs=rann_loss)
    else:

        output = Dense(1)(x)
        model = Model(inputs=x, outputs=output)

    return model
