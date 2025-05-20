import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    def __init__(self):
        self.data = None
        self.features = None
        self.labels = None
        self.results = None
        self.plots = {}

    def set_data(self, data, features=None, labels=None, results=None):
        """设置数据"""
        self.data = data
        self.features = features
        self.labels = labels
        self.results = results
        return True, "数据设置成功"

    def plot_time_series(self, x_column, y_column, title="时间序列图",
                         figsize=(12, 6), save_path=None):
        """绘制时间序列图"""
        if self.data is None or x_column not in self.data.columns or y_column not in self.data.columns:
            return False, "数据或列名不存在"

        plt.figure(figsize=figsize)
        plt.plot(self.data[x_column], self.data[y_column])
        plt.title(title)
        plt.xlabel(x_column)
        plt.ylabel(y_column)

        # 如果x是时间类型，设置日期格式
        if pd.api.types.is_datetime64_any_dtype(self.data[x_column]):
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.gcf().autofmt_xdate()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        self.plots[title] = plt.gcf()
        return True, plt.gcf()

    def plot_cluster_distribution(self, features=None, labels=None,
                                  method='pca', title="聚类分布图",
                                  figsize=(10, 8), save_path=None):
        """绘制聚类分布图"""
        if self.data is None:
            return False, "请先设置数据"

        if features is None and self.features is not None:
            features = self.features

        if labels is None and self.labels is not None:
            labels = self.labels

        if features is None or labels is None:
            return False, "缺少特征或标签数据"

        # 确保特征是DataFrame
        if isinstance(features, list):
            features_df = self.data[features]
        else:
            features_df = features

        # 降维处理
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            return False, f"不支持的降维方法: {method}"

        reduced_features = reducer.fit_transform(features_df)

        # 绘制散点图
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            reduced_features[:, 0],
            reduced_features[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.7,
            s=50
        )

        # 添加图例
        plt.colorbar(scatter, label='聚类标签')
        plt.title(title)
        plt.xlabel(f"{method.upper()} 第一主成分")
        plt.ylabel(f"{method.upper()} 第二主成分")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        self.plots[title] = plt.gcf()
        return True, plt.gcf()

    def plot_heatmap(self, columns=None, title="热力图", figsize=(12, 10),
                     save_path=None):
        """绘制热力图"""
        if self.data is None:
            return False, "请先设置数据"

        if columns is None:
            # 默认为所有数值列
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            return False, "没有找到数值列"

        # 计算相关系数矩阵
        corr = self.data[columns].corr()

        # 绘制热力图
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap='coolwarm', square=True, linewidths=.5,
                    cbar_kws={"shrink": .8})
        plt.title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        self.plots[title] = plt.gcf()
        return True, plt.gcf()

    def plot_radar_chart(self, data, labels, features, title="雷达图",
                         figsize=(10, 8), save_path=None):
        """绘制雷达图"""
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=features, index=labels)

        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()

        # 闭合雷达图
        features = features + [features[0]]
        angles = angles + [angles[0]]

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

        # 绘制每个类别的雷达图
        for i, label in enumerate(labels):
            values = data.iloc[i].tolist()
            values = values + [values[0]]  # 闭合
            ax.plot(angles, values, linewidth=2, label=label)
            ax.fill(angles, values, alpha=0.1)

        # 设置坐标轴
        ax.set_thetagrids(np.degrees(angles[:-1]), features)
        ax.set_ylim(0, data.values.max() * 1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        self.plots[title] = plt.gcf()
        return True, plt.gcf()

    def plot_health_index(self, health_index, time_column=None, title="健康指数曲线",
                          figsize=(12, 6), save_path=None):
        """绘制健康指数曲线"""
        if not isinstance(health_index, pd.Series) and not isinstance(health_index, pd.DataFrame):
            return False, "健康指数必须是pandas Series或DataFrame"

        plt.figure(figsize=figsize)

        if isinstance(health_index, pd.DataFrame):
            # 如果是DataFrame，假设第一列是时间列
            if time_column is None:
                time_column = health_index.columns[0]
                health_data = health_index.iloc[:, 1:]
            else:
                health_data = health_index.drop(columns=[time_column])

            # 绘制每条健康指数曲线
            for col in health_data.columns:
                plt.plot(health_index[time_column], health_data[col], label=col)
        else:
            # 如果是Series，假设索引是时间
            plt.plot(health_index.index, health_index)

        plt.title(title)
        plt.xlabel("时间")
        plt.ylabel("健康指数")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 如果有时间列，设置日期格式
        if time_column and pd.api.types.is_datetime64_any_dtype(health_index[time_column]):
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        self.plots[title] = plt.gcf()
        return True, plt.gcf()

    def plot_confusion_matrix(self, cm, class_names=None, title="混淆矩阵",
                              figsize=(10, 8), save_path=None):
        """绘制混淆矩阵"""
        if not isinstance(cm, np.ndarray):
            return False, "混淆矩阵必须是numpy数组"

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("预测类别")
        plt.ylabel("真实类别")
        plt.title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        self.plots[title] = plt.gcf()
        return True, plt.gcf()

    def generate_pdf_report(self, report_path, title="数据挖掘分析报告",
                            author="机电验证数据挖掘工具", plots=None):
        """生成PDF报告"""
        if plots is None:
            plots = self.plots

        if not plots:
            return False, "没有可用的图表来生成报告"

        try:
            with PdfPages(report_path) as pdf:
                # 添加标题页
                fig = plt.figure(figsize=(10, 8))
                fig.text(0.5, 0.7, title, ha='center', va='center', fontsize=24)
                fig.text(0.5, 0.6, f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                         ha='center', va='center', fontsize=14)
                fig.text(0.5, 0.5, f"作者: {author}", ha='center', va='center', fontsize=14)
                plt.axis('off')
                pdf.savefig()
                plt.close()

                # 添加每个图表
                for plot_title, plot_fig in plots.items():
                    pdf.savefig(plot_fig)
                    plt.close(plot_fig)

            return True, f"PDF报告已生成: {report_path}"
        except Exception as e:
            return False, f"生成PDF报告时出错: {str(e)}"