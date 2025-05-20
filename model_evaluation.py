import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 20})  # 设置字体大小


class ModelEvaluator:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.metrics = None

    def set_data(self, features, labels, test_size=0.2, random_state=42):
        """设置评估数据"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state
        )
        return True, "数据设置成功"

    def set_model(self, model):
        """设置评估模型"""
        self.model = model
        return True, "模型设置成功"

    def evaluate_classification_model(self, predict_proba=False):
        """评估分类模型"""
        if self.model is None or self.X_test is None or self.y_test is None:
            return False, "请先设置模型和数据"

        # 预测
        self.y_pred = self.model.predict(self.X_test)

        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred, average='weighted'),
            'recall': recall_score(self.y_test, self.y_pred, average='weighted'),
            'f1': f1_score(self.y_test, self.y_pred, average='weighted')
        }

        # 如果模型支持，计算AUC
        if predict_proba and hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(self.X_test)
            if y_prob.shape[1] > 2:  # 多分类问题
                metrics['auc'] = roc_auc_score(
                    self.y_test, y_prob, multi_class='ovr', average='weighted'
                )
            else:  # 二分类问题
                metrics['auc'] = roc_auc_score(self.y_test, y_prob[:, 1])

        # 混淆矩阵
        cm = confusion_matrix(self.y_test, self.y_pred)

        self.metrics = metrics

        return True, {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': classification_report(self.y_test, self.y_pred)
        }

    def evaluate_regression_model(self):
        """评估回归模型"""
        if self.model is None or self.X_test is None or self.y_test is None:
            return False, "请先设置模型和数据"

        # 预测
        self.y_pred = self.model.predict(self.X_test)

        # 计算评估指标
        metrics = {
            'mse': mean_squared_error(self.y_test, self.y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, self.y_pred)),
            'mae': mean_absolute_error(self.y_test, self.y_pred),
            'r2': r2_score(self.y_test, self.y_pred)
        }

        self.metrics = metrics

        return True, {
            'metrics': metrics,
            'y_true': self.y_test,
            'y_pred': self.y_pred
        }

    def perform_cross_validation(self, cv=5, scoring=None):
        """执行交叉验证"""
        if self.model is None or self.X_train is None or self.y_train is None:
            return False, "请先设置模型和训练数据"

        # 根据模型类型设置默认评分指标
        if scoring is None:
            if self._is_classification_model():
                scoring = 'accuracy'
            else:
                scoring = 'r2'

        # 执行交叉验证
        scores = cross_val_score(
            self.model, self.X_train, self.y_train, cv=cv, scoring=scoring
        )

        return True, {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }

    def optimize_hyperparameters(self, param_grid, cv=5, n_jobs=-1, method='grid'):
        """优化超参数"""
        if self.model is None or self.X_train is None or self.y_train is None:
            return False, "请先设置模型和训练数据"

        # 根据模型类型设置默认评分指标
        scoring = 'accuracy' if self._is_classification_model() else 'r2'

        # 选择优化方法
        if method == 'grid':
            search = GridSearchCV(
                self.model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=1
            )
        elif method == 'random':
            search = RandomizedSearchCV(
                self.model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs,
                verbose=1, n_iter=10
            )
        else:
            return False, f"不支持的优化方法: {method}"

        # 执行搜索
        search.fit(self.X_train, self.y_train)

        # 更新模型为最优模型
        self.model = search.best_estimator_

        return True, {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'search_results': pd.DataFrame(search.cv_results_)
        }

    def plot_confusion_matrix(self, normalize=False, title='混淆矩阵', figsize=(8, 6)):
        """绘制混淆矩阵"""
        if self.metrics is None or 'confusion_matrix' not in self.metrics:
            return False, "请先评估模型以获取混淆矩阵"

        cm = self.metrics['confusion_matrix']

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(title)

        return True, plt.gcf()

    def plot_prediction_comparison(self, title='预测值与真实值对比', figsize=(10, 6)):
        """绘制预测值与真实值对比图"""
        if self.metrics is None or 'y_true' not in self.metrics or 'y_pred' not in self.metrics:
            return False, "请先评估回归模型以获取预测值和真实值"

        y_true = self.metrics['y_true']
        y_pred = self.metrics['y_pred']

        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(title)

        return True, plt.gcf()

    def _is_classification_model(self):
        """判断模型是否为分类模型"""
        if hasattr(self.model, 'predict_proba') or hasattr(self.model, 'decision_function'):
            return True
        return False