import os
import numpy as np
import tensorflow as tf
import argparse
import time
from sklearn.utils import class_weight
import model
import hickle as hkl
import json

# 设置随机种子
SEED = 1
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 默认配置
DEFAULT_CONFIG = {
    "dataset": "MotionSense",
    "batch_size": 64,
    "epochs": 200,
    "learning_rate": 5e-3,
    "dropout_rate": 0.3,
    "segment_size": 128,
    "num_channels": 6,
}

# 活动标签映射
ACTIVITY_LABELS = {
    'MotionSense': ['Downstairs', 'Upstairs', 'Sitting', 'Standing', 'Walking', 'Jogging'],
    'UCI': ['Walking', 'Upstair','Downstair', 'Sitting', 'Standing', 'Lying'],
    'HHAR': ['Sitting', 'Standing', 'Walking', 'Upstair', 'Downstairs', 'Biking'],
    'RealWorld': ['Downstairs','Upstairs', 'Jumping','Lying', 'Running', 'Sitting', 'Standing', 'Walking'],
    'SHL': ['Standing','Walking','Runing','Biking','Car','Bus','Train','Subway']
}

# 添加数据集配置
DATASET_CONFIG = {
    'MotionSense': {
        'num_users': 24,
        'num_classes': 6
    },
    'UCI': {
        'num_users': None,  # UCI 使用不同的加载方式
        'num_classes': 6
    }
    # 其他数据集配置可以在这里添加
}

def parse_args():
    parser = argparse.ArgumentParser(description='Train MobileHART for mobile deployment')
    parser.add_argument('--dataset', type=str, default='MotionSense',
                      choices=['MotionSense', 'UCI', 'HHAR', 'RealWorld', 'SHL'],
                      help='Dataset to use for training')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_CONFIG['learning_rate'])
    return parser.parse_args()

def load_dataset(dataset_name):
    """加载数据集"""
    if dataset_name == 'MotionSense':
        # 直接从处理好的 .hkl 文件加载数据
        data_path = 'datasetStandardized/MotionSense'
        
        # 加载所有用户数据
        all_data = []
        all_labels = []
        num_users = DATASET_CONFIG[dataset_name]['num_users']
        for i in range(num_users):  # 使用配置中的用户数量
            user_data = hkl.load(f'{data_path}/UserData{i}.hkl')
            user_label = hkl.load(f'{data_path}/UserLabel{i}.hkl')
            all_data.append(user_data)
            all_labels.append(user_label)
        
        # 合并所有用户数据
        X = np.vstack(all_data)
        y = np.concatenate(all_labels)
        
        # 转换为 one-hot 编码
        y = tf.keras.utils.to_categorical(y, num_classes=6)
        
        # 分割训练集和测试集
        X_train = X[:int(0.8 * len(X))]
        X_test = X[int(0.8 * len(X)):]
        y_train = y[:int(0.8 * len(y))]
        y_test = y[int(0.8 * len(y)):]
        
    elif dataset_name == 'UCI':
        from datasets.DATA_UCI import load_data
        X_train, y_train, X_test, y_test = load_data()
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented yet")
    
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_test):
    """数据预处理"""
    # 标准化加速度数据
    mean_acc = np.mean(X_train[:,:,:3])
    std_acc = np.std(X_train[:,:,:3])
    X_train[:,:,:3] = (X_train[:,:,:3] - mean_acc) / std_acc
    X_test[:,:,:3] = (X_test[:,:,:3] - mean_acc) / std_acc
    
    # 标准化陀螺仪数据
    mean_gyro = np.mean(X_train[:,:,3:])
    std_gyro = np.std(X_train[:,:,3:])
    X_train[:,:,3:] = (X_train[:,:,3:] - mean_gyro) / std_gyro
    X_test[:,:,3:] = (X_test[:,:,3:] - mean_gyro) / std_gyro
    
    return X_train, X_test, {
        'mean_acc': mean_acc, 'std_acc': std_acc,
        'mean_gyro': mean_gyro, 'std_gyro': std_gyro
    }

def create_mobile_model(input_shape, num_classes):
    """创建轻量级MobileHART模型"""
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # 分离加速度和陀螺仪数据
    acc_input = tf.keras.layers.Lambda(lambda x: x[:,:,:3])(inputs)
    gyro_input = tf.keras.layers.Lambda(lambda x: x[:,:,3:])(inputs)
    
    # 创建基础模型
    base_model = model.HART(
        input_shape=input_shape,
        activityCount=num_classes,
        projection_dim=32,  # 减小投影维度
        patchSize=16,
        timeStep=16,
        num_heads=2,  # 减少注意力头数
        filterAttentionHead=2,
        convKernels=[3, 7, 15],  # 减少卷积核数量
        mlp_head_units=[256],  # 减小MLP单元
        dropout_rate=DEFAULT_CONFIG['dropout_rate']
    )
    
    return base_model

def convert_to_tflite(model, dataset_name, preprocessing_params):
    """转换为TFLite模型"""
    # 创建输出目录
    os.makedirs('mobile_models', exist_ok=True)
    
    # 转换模型
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    
    # 保存 TFLite 模型
    model_path = f'mobile_models/mobile_hart_{dataset_name.lower()}.tflite'
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    
    # 保存预处理参数为 JSON
    params_dict = {
        'mean_acc': float(preprocessing_params['mean_acc']),
        'std_acc': float(preprocessing_params['std_acc']),
        'mean_gyro': float(preprocessing_params['mean_gyro']),
        'std_gyro': float(preprocessing_params['std_gyro'])
    }
    
    params_path = f'mobile_models/preprocessing_params_{dataset_name.lower()}.json'
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(params_dict, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Preprocessing parameters saved to {params_path}")

def main():
    # 加载训练好的模型
    model_path = "./HART_Results/MobileHART_16frameLength_16TimeStep_192ProjectionSize_0.005LR/MotionSense/bestValcheckpoint.weights.h5"
    model = tf.keras.models.load_model(model_path)
    
    # 计算预处理参数
    preprocessing_params = {
        'mean_acc': float(np.mean(train_data[:,:,:3])),
        'std_acc': float(np.std(train_data[:,:,:3])),
        'mean_gyro': float(np.mean(train_data[:,:,3:])),
        'std_gyro': float(np.std(train_data[:,:,3:]))
    }
    
    # 转换并保存
    convert_to_tflite(model, "motionsense", preprocessing_params)

if __name__ == "__main__":
    main() 