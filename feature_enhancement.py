from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Input, Dense, Dropout
from keras.models import Model
import tensorflow as tf

def feature_extractor(input_shape):
    """
    构建特征提取器，包括四个卷积块，分别是卷积（Conv）、ReLU激活、批量归一化（BN）和最大池化（MP）层。
    """
    x = Input(shape=input_shape)
    
    # 卷积块1
    x1 = Conv2D(32, (3, 3), padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    
    # 卷积块2
    x2 = Conv2D(64, (3, 3), padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    
    # 卷积块3
    x3 = Conv2D(128, (3, 3), padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2))(x3)
    
    # 卷积块4
    x4 = Conv2D(256, (3, 3), padding='same')(x3)
    x4 = BatchNormalization()(x4)
    x4 = ReLU()(x4)
    x4 = MaxPooling2D(pool_size=(2, 2))(x4)
    
    return Model(inputs=x, outputs=x4)

def frequency_domain_enhancement(x, n_freq_components=128):
    """
    频域增强模块：采用1D-FFT将时序信号转换到频域，并增强关键频率分量。
    """
    # 将时域信号转换到频域
    fft_output = tf.signal.fft(tf.cast(x, tf.complex64))
    
    # 选择频域的前n个关键频率分量
    fft_output = fft_output[:, :, :, :n_freq_components]  # 裁剪fft_output为前128个频率分量
    
    # 引入频率核，增强频域的频率分量
    freq_kernel = tf.Variable(tf.random.normal([n_freq_components, 1]), trainable=True)
    
    # 将fft_output转换为float32，以匹配freq_kernel的数据类型
    fft_output = tf.cast(fft_output, tf.float32)
    
    # 确保freq_kernel和fft_output的维度匹配
    enhanced_freq = tf.matmul(fft_output, freq_kernel)
    
    # 逆变换回到时域
    enhanced_signal = tf.signal.ifft(enhanced_freq)
    
    return enhanced_signal


def moe_decomp(x, window_size=5):
    """
    MOEDecomp模块：使用多尺度滑动平均池化进行时序信号的趋势与周期分解。
    """
    # 平滑处理，提取长期变化趋势
    trend = tf.signal.hamming_window(window_size)
    trend_signal = tf.nn.conv1d(x, trend, stride=1, padding='SAME')
    
    # 提取周期项
    cycle_signal = x - trend_signal
    
    return trend_signal, cycle_signal

def feedforward(x, hidden_units=512, dropout_rate=0.5):
    """
    Feedforward模块：两层全连接映射，并应用激活函数和Dropout正则化。
    """
    # 第一层全连接
    x = Dense(hidden_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # 第二层全连接
    x = Dense(hidden_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    return x

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Input, Dense, Dropout, Layer, Concatenate
from keras.models import Model
import tensorflow as tf

# Focus模块
def focus_module(x, patch_size=3, n_freq_components=128):
    """
    Focus模块：增强特征图中的局部信息。
    """
    # 1. 切片操作和滑动窗口：假设 x 是 4D 特征图
    patches = tf.image.extract_patches(x, sizes=[1, patch_size, patch_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    patches = tf.reshape(patches, [-1, n_freq_components, patch_size, patch_size])
    
    # 2. 在每个窗口内选择四个相邻特征像素
    selected_patches = patches[:, :4, :, :]  # 假设选取前四个像素
    
    # 3. 使用Concat函数沿通道维度拼接
    concatenated = Concatenate(axis=-1)(selected_patches)
    
    # 4. 使用1x1卷积进行整形
    x1 = Conv2D(n_freq_components, (1, 1), activation='relu', padding='same')(concatenated)
    
    return x1

# Token Learner模块
def token_learner(x, num_tokens=128):
    """
    Token Learner模块：保持补丁内Token之间的空间顺序。
    """
    # 计算空间注意权重（假设 x 是 4D 时频特征图）
    attention_map = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    tokens = tf.reduce_mean(x * attention_map, axis=[1, 2])  # 计算每个Token的加权平均
    return tokens

# Token Transformer模块
def token_transformer(x, num_heads=8):
    """
    Token Transformer模块：捕捉Token之间的全局依赖关系。
    """
    # 使用多头自注意力机制
    attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    x = x + attention  # 残差连接
    return x

# Token Fuser模块
def token_fuser(x, tokens):
    """
    Token Fuser模块：将全局Token信息与局部特征图融合。
    """
    # 简单地通过连接操作将局部特征图和Token信息融合
    fused = Concatenate(axis=-1)([x, tokens])
    
    # 使用线性层和sigmoid激活进行融合
    fused = Dense(1, activation='sigmoid')(fused)
    
    return fused

# GDE模块：集成Focus、Token Learner、Token Transformer、Token Fuser
def gde_module(x, patch_size=3, n_freq_components=128, num_tokens=128, num_heads=8):
    """
    全局依赖编码器（GDE）：集成Focus、Token Learner、Token Transformer、Token Fuser模块。
    """
    # Focus模块
    x1 = focus_module(x, patch_size, n_freq_components)
    
    # Token Learner模块
    tokens = token_learner(x1, num_tokens)
    
    # Token Transformer模块
    transformed_tokens = token_transformer(tokens, num_heads)
    
    # Token Fuser模块
    output = token_fuser(x1, transformed_tokens)
    
    return output

def feature_enhancement_model(input_shape, n_freq_components=128, hidden_units=512, execute_modules=1):
    """
    集成特征提取与增强模块，包括卷积神经网络、频域增强编码器、GDE编码器等。
    """
    x = Input(shape=input_shape)
    
    # 特征提取模块
    if execute_modules != 1:
        feature_extractor_model = feature_extractor(input_shape)
        x1 = feature_extractor_model(x)
    else:
        x1 = x
    
    # 频域增强模块
    if execute_modules != 1:
        x2 = frequency_domain_enhancement(x1, n_freq_components)
    else:
        x2 = x1

    # GDE模块（集成Focus、Token Learner、Token Transformer、Token Fuser）
    if execute_modules != 1:
        x3 = gde_module(x2, patch_size=3, n_freq_components=n_freq_components)
    else:
        x3 = x2

    # 全连接模块（Feedforward）
    if execute_modules != 1:
        x4 = feedforward(x3, hidden_units=hidden_units)
    else:
        x4 = x3
    
    # 最终输出层
    final_output = Dense(1, activation='sigmoid')(x4)
    
    # 返回模型
    model = Model(inputs=x, outputs=[final_output])  # 返回最终输出
    return model



