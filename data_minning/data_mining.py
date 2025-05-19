import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib


class DataMiner:
    def __init__(self):
        self.data = None
        self.features = None
        self.labels = None
        self.model = None
        self.model_type = None

    def set_data(self, data, features=None, labels=None):
        """设置数据"""
        self.data = data

        if features is not None:
            self.features = data[features]

        if labels is not None:
            self.labels = data[labels]

        return True, "数据设置成功"

    def perform_kmeans_clustering(self, n_clusters=3, max_iter=300):
        """执行K-Means聚类"""
        if self.features is None:
            return False, "请先设置特征数据"

        # 数据标准化
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.features)

        # 执行K-Means
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)

        self.model = kmeans
        self.model_type = "kmeans"

        return True, {
            'clusters': clusters,
            'inertia': kmeans.inertia_,
            'centers': kmeans.cluster_centers_
        }

    def perform_dbscan(self, eps=0.5, min_samples=5):
        """执行DBSCAN密度聚类"""
        if self.features is None:
            return False, "请先设置特征数据"

        # 数据标准化
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.features)

        # 执行DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_features)

        self.model = dbscan
        self.model_type = "dbscan"

        return True, {
            'clusters': clusters,
            'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
            'n_noise': list(clusters).count(-1)
        }

    def perform_spectral_clustering(self, n_clusters=3):
        """执行谱聚类"""
        if self.features is None:
            return False, "请先设置特征数据"

        # 数据标准化
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.features)

        # 执行谱聚类
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
        clusters = spectral.fit_predict(scaled_features)

        self.model = spectral
        self.model_type = "spectral"

        return True, {
            'clusters': clusters
        }

    def detect_anomalies_isolation_forest(self, contamination=0.05):
        """使用孤立森林进行异常检测"""
        if self.features is None:
            return False, "请先设置特征数据"

        # 数据标准化
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.features)

        # 执行孤立森林
        model = IsolationForest(contamination=contamination, random_state=42)
        anomalies = model.fit_predict(scaled_features)

        # 将-1(异常)转换为1，1(正常)转换为0
        anomalies = np.where(anomalies == -1, 1, 0)

        self.model = model
        self.model_type = "isolation_forest"

        return True, {
            'anomalies': anomalies,
            'anomaly_scores': -model.decision_function(scaled_features)
        }

    def detect_anomalies_one_class_svm(self, nu=0.1, kernel='rbf'):
        """使用One-Class SVM进行异常检测"""
        if self.features is None:
            return False, "请先设置特征数据"

        # 数据标准化
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.features)

        # 执行One-Class SVM
        model = OneClassSVM(nu=nu, kernel=kernel)
        anomalies = model.fit_predict(scaled_features)

        # 将-1(异常)转换为1，1(正常)转换为0
        anomalies = np.where(anomalies == -1, 1, 0)

        self.model = model
        self.model_type = "one_class_svm"

        return True, {
            'anomalies': anomalies
        }

    def detect_anomalies_lof(self, n_neighbors=20):
        """使用局部离群因子(LOF)进行异常检测"""
        if self.features is None:
            return False, "请先设置特征数据"

        # 数据标准化
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.features)

        # 执行LOF
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
        anomalies = model.fit_predict(scaled_features)

        # 将-1(异常)转换为1，1(正常)转换为0
        anomalies = np.where(anomalies == -1, 1, 0)

        self.model = model
        self.model_type = "lof"

        return True, {
            'anomalies': anomalies,
            'outlier_scores': -model.negative_outlier_factor_
        }

    def train_regression_model(self, model_type='random_forest', test_size=0.2, params=None):
        """训练回归模型"""
        if self.features is None or self.labels is None:
            return False, "请先设置特征数据和标签"

        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=test_size, random_state=42
        )

        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 选择模型
        if model_type == 'random_forest':
            if params is None:
                params = {'n_estimators': 100, 'random_state': 42}
            model = RandomForestRegressor(**params)
        elif model_type == 'gradient_boosting':
            if params is None:
                params = {'n_estimators': 100, 'random_state': 42}
            model = GradientBoostingRegressor(**params)
        elif model_type == 'svr':
            if params is None:
                params = {'kernel': 'rbf', 'C': 1.0}
            model = SVR(**params)
        else:
            return False, f"不支持的模型类型: {model_type}"

        # 训练模型
        model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_test_scaled)

        # 评估
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        self.model = model
        self.model_type = model_type

        return True, {
            'model': model,
            'metrics': metrics,
            'y_true': y_test,
            'y_pred': y_pred
        }

    def save_model(self, file_path):
        """保存模型"""
        if self.model is None:
            return False, "没有训练好的模型可保存"

        try:
            joblib.dump(self.model, file_path)
            return True, f"模型已保存至: {file_path}"
        except Exception as e:
            return False, f"模型保存失败: {str(e)}"

    def load_model(self, file_path):
        """加载模型"""
        try:
            self.model = joblib.load(file_path)
            return True, "模型加载成功"
        except Exception as e:
            return False, f"模型加载失败: {str(e)}"

    def predict(self, data):
        """使用模型进行预测"""
        if self.model is None:
            return False, "没有可用的模型"

        # 数据标准化
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # 预测
        predictions = self.model.predict(data_scaled)

        return True, predictions