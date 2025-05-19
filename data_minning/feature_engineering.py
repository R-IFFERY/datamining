import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso


class FeatureEngineer:
    def __init__(self):
        self.data = None
        self.features = None
        self.labels = None
        self.selected_features = None

    def set_data(self, data, labels=None):
        """设置数据"""
        self.data = data
        self.labels = labels

        # 假设所有数值列都是特征
        if self.features is None:
            self.features = data.select_dtypes(include=[np.number])

        return True, "数据设置成功"

    def extract_time_domain_features(self, column, window_size=100, step=50):
        """提取时域特征"""
        if column not in self.data.columns:
            return False, f"列 {column} 不存在"

        signal = self.data[column].values
        features = []

        for i in range(0, len(signal) - window_size + 1, step):
            window = signal[i:i + window_size]
            window_features = {
                'mean': np.mean(window),
                'std': np.std(window),
                'var': np.var(window),
                'max': np.max(window),
                'min': np.min(window),
                'range': np.max(window) - np.min(window),
                'skewness': stats.skew(window),
                'kurtosis': stats.kurtosis(window),
                'rms': np.sqrt(np.mean(window ** 2)),
                'crest_factor': np.max(window) / np.sqrt(np.mean(window ** 2))
            }
            features.append(window_features)

        return True, pd.DataFrame(features)

    def extract_frequency_domain_features(self, column, window_size=100, step=50):
        """提取频域特征"""
        if column not in self.data.columns:
            return False, f"列 {column} 不存在"

        signal = self.data[column].values
        features = []

        for i in range(0, len(signal) - window_size + 1, step):
            window = signal[i:i + window_size]
            # 应用FFT
            fft = np.fft.fft(window)
            fft_freq = np.fft.fftfreq(len(fft))
            # 只取正频率部分
            n = len(fft)
            fft_pos = fft[:n // 2]
            fft_freq_pos = fft_freq[:n // 2]
            fft_mag = np.abs(fft_pos)

            # 计算频域特征
            total_power = np.sum(fft_mag ** 2)
            if total_power == 0:
                total_power = 1e-10  # 避免除零错误

            dominant_freq = fft_freq_pos[np.argmax(fft_mag)]
            spectral_centroid = np.sum(fft_freq_pos * fft_mag) / np.sum(fft_mag)
            bandwidth = np.sqrt(np.sum(fft_freq_pos ** 2 * fft_mag) / np.sum(fft_mag) - spectral_centroid ** 2)

            window_features = {
                'dominant_frequency': dominant_freq,
                'spectral_centroid': spectral_centroid,
                'bandwidth': bandwidth,
                'total_power': total_power,
                'frequency_spread': np.std(fft_freq_pos * fft_mag)
            }
            features.append(window_features)

        return True, pd.DataFrame(features)

    def select_features_by_variance(self, threshold=0.01):
        """基于方差过滤特征"""
        if self.features is None:
            return False, "请先设置特征数据"

        variances = self.features.var()
        selected = variances[variances > threshold].index.tolist()

        self.selected_features = self.features[selected]
        return True, f"已选择 {len(selected)} 个特征，原始特征数: {self.features.shape[1]}"

    def select_features_by_mutual_info(self, k=10):
        """基于互信息选择特征"""
        if self.features is None or self.labels is None:
            return False, "请先设置特征数据和标签"

        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(self.features, self.labels)
        selected_indices = selector.get_support(indices=True)
        selected = self.features.columns[selected_indices].tolist()

        self.selected_features = self.features[selected]
        return True, f"已选择 {len(selected)} 个特征，原始特征数: {self.features.shape[1]}"

    def select_features_by_random_forest(self, n_estimators=100, k=10):
        """基于随机森林重要性选择特征"""
        if self.features is None or self.labels is None:
            return False, "请先设置特征数据和标签"

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(self.features, self.labels)

        # 获取特征重要性
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # 选择前k个特征
        selected = self.features.columns[indices[:k]].tolist()

        self.selected_features = self.features[selected]
        return True, f"已选择 {len(selected)} 个特征，原始特征数: {self.features.shape[1]}"

    def perform_pca(self, n_components=0.95):
        """执行PCA降维"""
        if self.features is None:
            return False, "请先设置特征数据"

        # 数据标准化
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.features)

        # 执行PCA
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(scaled_features)

        # 创建包含PCA成分的DataFrame
        pca_df = pd.DataFrame(
            data=pca_features,
            columns=[f'PC{i + 1}' for i in range(pca_features.shape[1])]
        )

        self.selected_features = pca_df
        return True, {
            'message': f"PCA降维完成，保留了{n_components * 100:.1f}%的方差",
            'explained_variance': pca.explained_variance_ratio_,
            'n_components': pca.n_components_
        }

    def select_features_by_lasso(self, alpha=0.01):
        """基于LASSO回归选择特征"""
        if self.features is None or self.labels is None:
            return False, "请先设置特征数据和标签"

        # 数据标准化
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.features)

        # 执行LASSO
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(scaled_features, self.labels)

        # 获取非零系数的特征
        selected_indices = np.where(lasso.coef_ != 0)[0]
        selected = self.features.columns[selected_indices].tolist()

        self.selected_features = self.features[selected]
        return True, f"已选择 {len(selected)} 个特征，原始特征数: {self.features.shape[1]}"

    def get_feature_importance(self):
        """获取特征重要性评分"""
        if self.selected_features is None:
            return False, "请先选择特征"

        # 这里使用随机森林计算特征重要性作为示例
        if self.labels is not None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(self.selected_features, self.labels)

            importance_df = pd.DataFrame({
                'Feature': self.selected_features.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            return True, importance_df
        else:
            return False, "需要标签数据来计算特征重要性"