import pandas as pd
import numpy as np
from scipy import stats
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

class DataProcessor:
    """数据预处理模块"""

    def __init__(self):
        self.data = None
        self.file_path = None
        self.file_format = None
        self.processed_data = None
        self.original_shape = None
        self.current_shape = None

    def load_data(self, file_path, file_format='.csv', **kwargs):
        """加载数据文件

        Args:
            file_path (str): 文件路径
            file_format (str): 文件格式，支持 'csv', 'excel', 'txt'
            **kwargs: 传递给pandas读取函数的参数

        Returns:
            tuple: (成功标志, 消息)
        """
        try:
            self.file_path = file_path
            self.file_format = file_format

            if file_format.lower() == '.csv':
                self.data = pd.read_csv(file_path, **kwargs)
            elif file_format.lower() in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path, **kwargs)
            elif file_format.lower() == '.txt':
                self.data = pd.read_table(file_path, **kwargs)
            else:
                return False, f"不支持的文件格式: {file_format}"

            self.processed_data = self.data.copy()
            self.original_shape = self.data.shape
            self.current_shape = self.data.shape

            return True, f"数据加载成功，共{self.data.shape[0]}行，{self.data.shape[1]}列"
        except Exception as e:
            return False, f"加载数据时出错: {str(e)}"

    def remove_duplicates(self):
        """去除重复记录

        Returns:
            tuple: (成功标志, 消息)
        """
        try:
            if self.processed_data is None:
                return False, "没有加载的数据可处理"

            original_rows = self.processed_data.shape[0]
            self.processed_data.drop_duplicates(inplace=True)
            removed_rows = original_rows - self.processed_data.shape[0]
            self.current_shape = self.processed_data.shape

            return True, f"已移除{removed_rows}条重复记录"
        except Exception as e:
            return False, f"去除重复记录时出错: {str(e)}"

    def handle_missing_values(self, method='mean'):
        """处理缺失值

        Args:
            method (str): 处理方法，支持 'mean'（均值）, 'median'（中位数）,
                          'interpolate'（插值）

        Returns:
            tuple: (成功标志, 消息)
        """
        try:
            if self.processed_data is None:
                return False, "没有加载的数据可处理"

            missing_count = self.processed_data.isnull().sum().sum()

            if missing_count == 0:
                return True, "数据中没有缺失值"

            if method == '均值插补':
                # 使用均值填充
                for column in self.processed_data.select_dtypes(include=[np.number]).columns:
                    self.processed_data[column].fillna(self.processed_data[column].mean(), inplace=True)
            elif method == '中位数插补':
                # 使用中位数填充
                for column in self.processed_data.select_dtypes(include=[np.number]).columns:
                    self.processed_data[column].fillna(self.processed_data[column].median(), inplace=True)
            elif method == '时间序列插值':
                # 使用插值法填充
                for column in self.processed_data.select_dtypes(include=[np.number]).columns:
                    self.processed_data[column] = self.processed_data[column].interpolate()
            else:
                return False, f"不支持的缺失值处理方法: {method}"

            new_missing_count = self.processed_data.isnull().sum().sum()
            self.current_shape = self.processed_data.shape

            return True, f"已处理{missing_count - new_missing_count}个缺失值"
        except Exception as e:
            return False, f"处理缺失值时出错: {str(e)}"

    def detect_outliers(self, method='zscore', threshold=3.0):
        """检测并处理异常值

        Args:
            method (str): 检测方法，支持 'zscore'（Z-score）, 'iqr'（四分位距）,
                          'isolation_forest'（孤立森林）, 'One-Class SVM'（单类SVM）
            threshold (float): 阈值，用于判断异常值的临界值

        Returns:
            tuple: (成功标志, 消息)
        """
        try:
            if self.processed_data is None:
                return False, "没有加载的数据可处理"

            numeric_columns = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_columns:
                return False, "数据中没有数值类型的列"

            outlier_count = 0

            if method == 'Z-score':
                # 使用Z-score方法检测异常值
                for column in numeric_columns:
                    z_scores = np.abs(stats.zscore(self.processed_data[column]))
                    outliers = z_scores > threshold
                    outlier_count += outliers.sum()

                    # 可以选择替换或删除异常值，这里选择替换为NaN
                    self.processed_data.loc[outliers, column] = np.nan

            elif method == 'IQR方法':
                # 使用IQR方法检测异常值
                for column in numeric_columns:
                    Q1 = self.processed_data[column].quantile(0.25)
                    Q3 = self.processed_data[column].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR

                    outliers = (self.processed_data[column] < lower_bound) | (self.processed_data[column] > upper_bound)
                    outlier_count += outliers.sum()

                    # 替换为NaN
                    self.processed_data.loc[outliers, column] = np.nan

            elif method == '孤立森林':
                # 使用Isolation Forest方法检测异常值
                for column in numeric_columns:
                    iso = IsolationForest(contamination=threshold)
                    outliers = iso.fit_predict(self.processed_data[[column]]) == -1
                    outlier_count += outliers.sum()

                    # 替换为NaN
                    self.processed_data.loc[outliers, column] = np.nan

            elif method == 'One-Class SVM':
                # 使用One-Class SVM方法检测异常值
                for column in numeric_columns:
                    svm = OneClassSVM(contamination=threshold)
                    outliers = svm.fit_predict(self.processed_data[[column]]) == -1
                    outlier_count += outliers.sum()

                    # 替换为NaN
                    self.processed_data.loc[outliers, column] = np.nan

            else:
                return False, f"不支持的异常值检测方法: {method}"

            return True, f"已检测到{outlier_count}个异常值，已将其标记为缺失值"
        except Exception as e:
            return False, f"检测异常值时出错: {str(e)}"

    def normalize_data(self, method='standard'):
        """标准化/归一化数据

        Args:
            method (str): 处理方法，支持 'standard'（Z-score标准化）,
                          'minmax'（Min-Max归一化）, 'robust'（RobustScaler）,
                          'normalizer'（Normalizer）

        Returns:
            tuple: (成功标志, 消息)
        """
        try:
            if self.processed_data is None:
                return False, "没有加载的数据可处理"

            numeric_columns = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_columns:
                return False, "数据中没有数值类型的列"

            if method == 'Z-score标准化':
                scaler = StandardScaler()
            elif method == 'Min-Max归一化':
                scaler = MinMaxScaler()
            elif method == 'RobustScaler':
                scaler = RobustScaler()
            elif method == 'Normalizer':
                scaler = Normalizer()
            else:
                return False, f"不支持的标准化/归一化方法: {method}"

            # 应用标准化/归一化
            self.processed_data[numeric_columns] = scaler.fit_transform(self.processed_data[numeric_columns])

            return True, f"已对{len(numeric_columns)}列数据进行{method}标准化/归一化"
        except Exception as e:
            return False, f"标准化/归一化数据时出错: {str(e)}"

    def preview_result(self):
        """预览处理后的数据

        Returns:
            pd.DataFrame: 处理后的数据
        """
        if self.processed_data is None:
            return None

        # 返回前5行数据
        return self.processed_data.head(5)

    def save_processed_data(self, file_path, file_format='.csv', **kwargs):
        """保存处理后的数据

        Args:
            file_path (str): 保存文件路径
            file_format (str): 文件格式，支持 'csv', 'excel', 'txt'
            **kwargs: 传递给pandas写入函数的参数

        Returns:
            tuple: (成功标志, 消息)
        """
        try:
            if self.processed_data is None:
                return False, "没有处理后的数据可保存"

            if file_format.lower() == '.csv':
                self.processed_data.to_csv(file_path, **kwargs)
            elif file_format.lower() in ['.xlsx', '.xls']:
                self.processed_data.to_excel(file_path, **kwargs)
            elif file_format.lower() == '.txt':
                self.processed_data.to_csv(file_path, sep='\t', **kwargs)
            else:
                return False, f"不支持的文件格式: {file_format}"

            return True, f"处理后的数据已保存至 {file_path}"
        except Exception as e:
            return False, f"保存处理后的数据时出错: {str(e)}"

    def get_data_summary(self):
        """获取数据的基本统计信息"""
        if self.processed_data is None:
            return None
        return self.processed_data.describe()