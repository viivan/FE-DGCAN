from keras.models import Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization,Reshape
from keras.optimizers import Adam
import tensorflow as tf


def vae_pretraining(input_shape, latent_dim, learning_rate=1.5e-4):
    """
    VAE预训练阶段：最小化重构误差与KL散度
    """
    # 编码器
    x = Input(shape=input_shape)
    encoded = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(latent_dim, activation='relu')(encoded)

    # 解码器
    decoded = Dense(input_shape[0] * input_shape[1], activation='sigmoid')(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Reshape(input_shape)(decoded)

    # VAE模型
    vae = Model(x, decoded)
    
    # 使用重构误差和KL散度作为损失函数
    def vae_loss(y_true, y_pred):
        reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        kl_loss = -0.5 * tf.reduce_mean(1 + encoded - tf.square(encoded) - tf.exp(encoded))
        return reconstruction_loss + kl_loss
    
    vae.compile(optimizer=Adam(learning_rate), loss=vae_loss)
    return vae

def contrastive_loss(x1, x2, margin=1.0):
    """
    无监督对比损失：增强源域和目标域特征一致性与判别性
    """
    euclidean_distance = tf.norm(x1 - x2, axis=-1)
    return tf.reduce_mean(tf.maximum(euclidean_distance - margin, 0))

def unsupervised_contrastive_learning(x_source, x_target, model, margin=1.0):
    """
    无监督对比分类学习：训练源域与目标域特征
    """
    # 提取源域和目标域特征
    source_features = model(x_source)
    target_features = model(x_target)

    # 计算对比损失
    loss = contrastive_loss(source_features, target_features, margin)
    return loss
def rann_loss(x_source, x_target, model, alpha=0.5):
    """
    残差对齐神经网络：对源域与目标域特征进行对齐
    """
    # 获取源域和目标域的特征
    source_features = model(x_source)
    target_features = model(x_target)
    
    # 计算源域和目标域的协方差矩阵
    source_cov = tf.matmul(source_features, source_features, transpose_a=True)
    target_cov = tf.matmul(target_features, target_features, transpose_a=True)
    
    # 计算CORAL损失
    coral_loss = tf.reduce_mean(tf.square(source_cov - target_cov))

    # 计算对比损失和分类损失
    contrastive_loss_val = contrastive_loss(source_features, target_features)
    
    # 总损失：对比损失 + CORAL损失
    total_loss = alpha * coral_loss + (1 - alpha) * contrastive_loss_val
    
    return total_loss

def anomaly_detection(x, model, threshold_factor=3.0):
    """
    异常检测：通过重构误差进行异常判定
    """
    # 通过VAE或其他编码器获取潜在特征
    latent_features = model.encode(x)
    
    # 使用模型重构输入信号
    reconstructed_x = model.decode(latent_features)
    
    # 计算重构误差：均方误差
    reconstruction_error = tf.reduce_mean(tf.square(x - reconstructed_x), axis=-1)
    
    # 计算源域正常样本的重构误差分布，确定动态阈值
    mean_reconstruction_error = tf.reduce_mean(reconstruction_error)
    std_reconstruction_error = tf.math.reduce_std(reconstruction_error)
    dynamic_threshold = mean_reconstruction_error + threshold_factor * std_reconstruction_error
    
    # 判断样本是否为异常：若重构误差大于阈值，则为异常
    anomaly_scores = reconstruction_error > dynamic_threshold
    return anomaly_scores


def train(model, x_source, x_target, epochs=200, execute_modules=1):
    """
    训练过程：包括预训练、无监督对比学习和异常检测
    如果 execute_modules 参数是 1，则跳过对应模块的执行。
    
    execute_modules：1 表示不执行模块，其他值表示执行模块
    """
    # 预训练阶段
    if execute_modules != 1:  # 如果不是 1，执行预训练
        vae = vae_pretraining(input_shape=x_source.shape[1:], latent_dim=128)
        vae.fit(x_source, x_source, epochs=epochs)

    # 无监督对比学习阶段
    if execute_modules != 1:  # 如果不是 1，执行无监督对比学习
        for epoch in range(epochs):
            contrastive_loss_value = unsupervised_contrastive_learning(x_source, x_target, model)
            print(f'Epoch {epoch} - Contrastive Loss: {contrastive_loss_value}')

    # 异常检测阶段
    if execute_modules != 1:  # 如果不是 1，执行异常检测
        anomaly_scores = anomaly_detection(x_target, model)
        print(f'Anomaly Scores: {anomaly_scores}')
    
    return model
