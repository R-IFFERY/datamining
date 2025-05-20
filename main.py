import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox, QTableWidgetItem, QTableWidget
from PyQt5.QtCore import Qt
from main_window import MainWindow  # 假设界面代码在main_window.py中
from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from data_mining import DataMiner
from model_evaluation import ModelEvaluator
from visualization import Visualizer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MplCanvas(FigureCanvas):
    """Matplotlib画布，用于在PyQt5中显示图表"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()



class AppController:
    def __init__(self):
        # 创建应用和主窗口
        self.app = QApplication(sys.argv)
        self.window = MainWindow()

        # 创建功能模块实例
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.data_miner = DataMiner()
        self.model_evaluator = ModelEvaluator()
        self.visualizer = Visualizer()

        # 连接信号和槽
        self.connect_signals()

    def connect_signals(self):
        """连接界面按钮与功能函数"""
        # 数据预处理标签页
        self.window.execute_preprocessing_btn.clicked.connect(self.execute_data_preprocessing)
        self.window.preview_result_btn.clicked.connect(self.preview_data_result)
        self.window.save_processed_data_btn.clicked.connect(self.save_processed_data)

        # 特征工程标签页
        self.window.extract_features_btn.clicked.connect(self.extract_features)
        self.window.select_features_btn.clicked.connect(self.select_features)

        # 数据挖掘标签页
        self.window.run_algorithm_btn.clicked.connect(self.run_data_mining_algorithm)
        self.window.save_model_btn.clicked.connect(self.save_model)

        # 模型评估标签页
        self.window.evaluate_btn.clicked.connect(self.evaluate_model)
        self.window.compare_btn.clicked.connect(self.compare_models)
        self.window.export_report_btn.clicked.connect(self.export_evaluation_report)

        # 可视化标签页
        self.window.refresh_chart_btn.clicked.connect(self.refresh_chart)
        self.window.export_chart_btn.clicked.connect(self.export_chart)
        self.window.generate_report_btn.clicked.connect(self.generate_report)

    def execute_data_preprocessing(self):
        """执行数据预处理"""
        try:
            file_path = self.window.file_path_edit.text()
            if self.window.csv_radio.isChecked():
                file_format = ".csv"
            elif self.window.excel_radio.isChecked():
                file_format = ".xlsx"
            elif self.window.txt_radio.isChecked():
                file_format = ".txt"

            # 加载数据
            success, message = self.data_processor.load_data(file_path, file_format)
            if not success:
                self.show_error_message("数据加载错误", message)
                return

            # 去除重复值
            if self.window.duplicate_check.isChecked():
                success, message = self.data_processor.remove_duplicates()
                if not success:
                    self.show_error_message("数据处理错误", message)
                    return

            # 处理缺失值
            if self.window.missing_check.isChecked():
                method = self.window.missing_method_combo.currentText()
                success, message = self.data_processor.handle_missing_values(method)
                if not success:
                    self.show_error_message("数据处理错误", message)
                    return

            # 检测异常值
            if self.window.outlier_check.isChecked():
                method = self.window.outlier_method_combo.currentText()
                threshold = float(self.window.outlier_threshold_edit.text())
                success, message = self.data_processor.detect_outliers(method, threshold)
                if not success:
                    self.show_error_message("数据处理错误", message)
                    return

            # 标准化/归一化
            if self.window.normalization_check.isChecked():
                method = self.window.normalization_method_combo.currentText()
                success, message = self.data_processor.normalize_data(method)
                if not success:
                    self.show_error_message("数据处理错误", message)
                    return

            self.show_info_message("成功", "数据预处理完成")
            self.window.statusBar().showMessage("数据预处理完成")

            # 设置特征工程的数据
            self.feature_engineer.set_data(self.data_processor.processed_data)
            # 设置可视化数据
            self.visualizer.set_data(self.data_processor.processed_data)

        except Exception as e:
            self.show_error_message("执行错误", str(e))

    def preview_data_result(self):
        """预览数据处理结果"""
        try:
            if self.data_processor.processed_data is None:
                self.show_error_message("错误", "没有处理后的数据可预览")
                return

            # 弹出界面显示get_data_summary中的数据
            summary = self.data_processor.get_data_summary()
            # 这里需要根据实际界面设计实现数据预览功能

        except Exception as e:
            self.show_error_message("预览错误", str(e))

    def save_processed_data(self):
        """保存处理后的数据"""
        try:
            file_path, _ = self.window.get_save_file_path()
            if not file_path:
                return

            if self.window.csv_radio.isChecked():
                file_format = ".csv"
            elif self.window.excel_radio.isChecked():
                file_format = ".xlsx"
            elif self.window.txt_radio.isChecked():
                file_format = ".txt"

            success, message = self.data_processor.save_processed_data(file_path, file_format)
            if success:
                self.show_info_message("成功", message)
            else:
                self.show_error_message("保存错误", message)
        except Exception as e:
            self.show_error_message("保存错误", str(e))

    def extract_features(self):
        """提取特征"""
        try:
            if self.data_processor.processed_data is None:
                self.show_error_message("错误", "请先完成数据预处理")
                return

            # 获取特征提取参数
            feature_type = None
            for key, radio in self.window.feature_type_radios.items():
                if radio.isChecked():
                    feature_type = key
                    break

            columns = self.data_processor.processed_data.select_dtypes(include=[np.number]).columns.tolist()

            if "时域" in feature_type:
                # 提取时域特征
                window_size = int(self.window.window_size_edit.text())
                step_size = int(self.window.step_size_edit.text())

                for col in columns:
                    success, features = self.feature_engineer.extract_time_domain_features(
                        col, window_size, step_size
                    )
                    if not success:
                        self.show_error_message("特征提取错误", features)
                        return

            elif "频域" in feature_type:
                # 提取频域特征
                window_size = int(self.window.window_size_edit.text())
                step_size = int(self.window.step_size_edit.text())

                for col in columns:
                    success, features = self.feature_engineer.extract_frequency_domain_features(
                        col, window_size, step_size
                    )
                    if not success:
                        self.show_error_message("特征提取错误", features)
                        return

            self.show_info_message("成功", "特征提取完成")
            self.window.statusBar().showMessage("特征提取完成")

        except Exception as e:
            self.show_error_message("特征提取错误", str(e))

    def select_features(self):
        """选择特征"""
        try:
            if self.feature_engineer.features is None:
                self.show_error_message("错误", "请先提取特征")
                return

            # 获取特征选择方法
            method = None
            for key, check in self.window.selection_method_checks.items():
                if check.isChecked():
                    method = key
                    break

            if "方差过滤法" in method:
                threshold = float(self.window.variance_threshold_edit.text())
                success, message = self.feature_engineer.select_features_by_variance(threshold)
            elif "互信息与相关性分析" in method:
                k = int(self.window.k_features_edit.text())
                success, message = self.feature_engineer.select_features_by_mutual_info(k)
            elif "基于模型的特征重要性排序" in method:
                k = int(self.window.k_features_edit.text())
                n_estimators = int(self.window.n_estimators_edit.text())
                success, message = self.feature_engineer.select_features_by_random_forest(n_estimators, k)
            elif "PCA主成分分析" in method:
                n_components = float(self.window.pca_components_edit.text())
                success, result = self.feature_engineer.perform_pca(n_components)
                message = result['message']
            elif "LASSO稀疏回归" in method:
                alpha = float(self.window.lasso_alpha_edit.text())
                success, message = self.feature_engineer.select_features_by_lasso(alpha)
            else:
                self.show_error_message("错误", f"不支持的特征选择方法: {method}")
                return

            if success:
                self.show_info_message("成功", message)
                self.window.statusBar().showMessage("特征选择完成")
            else:
                self.show_error_message("特征选择错误", message)

        except Exception as e:
            self.show_error_message("特征选择错误", str(e))

    def run_data_mining_algorithm(self):
        """运行数据挖掘算法"""
        try:
            if self.feature_engineer.selected_features is None:
                self.show_error_message("错误", "请先完成特征工程")
                return

            # 获取算法类型
            algorithm_type = ""
            # 从界面获取算法类型

            # 设置数据
            self.data_miner.set_data(
                self.data_processor.data,
                self.feature_engineer.selected_features.columns.tolist(),
                self.window.target_column_edit.text()
            )

            if "聚类" in algorithm_type:
                # 聚类算法
                n_clusters = int(self.window.n_clusters_spin.value())
                max_iter = int(self.window.max_iter_spin.value())

                success, result = self.data_miner.perform_kmeans_clustering(n_clusters, max_iter)
                if success:
                    self.show_info_message("成功", f"聚类完成，共{result['n_clusters']}个聚类")
                else:
                    self.show_error_message("聚类错误", result)

            elif "异常检测" in algorithm_type:
                # 异常检测算法
                method = self.window.anomaly_method_combo.currentText()
                contamination = float(self.window.contamination_edit.text())

                success, result = self.data_miner.detect_anomalies_isolation_forest(contamination)
                if success:
                    self.show_info_message("成功", "异常检测完成")
                else:
                    self.show_error_message("异常检测错误", result)

            # 训练回归模型
            elif "回归" in algorithm_type:
                model_type = self.window.regression_model_combo.currentText()
                success, result = self.data_miner.train_regression_model(model_type)
                if success:
                    self.show_info_message("成功", "回归模型训练完成")
                    # 设置评估模型和数据
                    self.model_evaluator.set_model(self.data_miner.model)
                    self.model_evaluator.set_data(
                        self.data_miner.features,
                        self.data_miner.labels
                    )
                else:
                    self.show_error_message("回归模型训练错误", result)

        except Exception as e:
            self.show_error_message("数据挖掘错误", str(e))

    def save_model(self):
        """保存模型"""
        try:
            file_path, _ = self.window.get_save_file_path()
            if not file_path:
                return

            success, message = self.data_miner.save_model(file_path)
            if success:
                self.show_info_message("成功", message)
            else:
                self.show_error_message("保存错误", message)
        except Exception as e:
            self.show_error_message("保存错误", str(e))

    def evaluate_model(self):
        """评估模型"""
        try:
            if self.model_evaluator.model is None:
                self.show_error_message("错误", "请先训练模型")
                return

            if self.model_evaluator._is_classification_model():
                success, result = self.model_evaluator.evaluate_classification_model(predict_proba=True)
            else:
                success, result = self.model_evaluator.evaluate_regression_model()

            if success:
                self.show_info_message("成功", "模型评估完成")
                # 可以在这里添加显示评估结果的逻辑
            else:
                self.show_error_message("评估错误", result)
        except Exception as e:
            self.show_error_message("评估错误", str(e))

    def compare_models(self):
        """比较多个模型"""
        try:
            # 这里实现模型比较功能
            self.show_info_message("提示", "模型比较功能正在开发中")
        except Exception as e:
            self.show_error_message("比较错误", str(e))

    def export_evaluation_report(self):
        """导出评估报告"""
        try:
            file_path, _ = self.window.get_save_file_path()
            if not file_path:
                return

            # 生成报告
            success, message = self.model_evaluator.generate_report(file_path)
            if success:
                self.show_info_message("成功", message)
            else:
                self.show_error_message("导出错误", message)
        except Exception as e:
            self.show_error_message("导出错误", str(e))

    def refresh_chart(self):
        """刷新图表"""
        try:
            chart_type = self.window.chart_type_combo.currentText()

            if "趋势图" in chart_type:
                x_column = self.window.x_column_combo.currentText()
                y_column = self.window.y_column_combo.currentText()
                success, fig = self.visualizer.plot_time_series(x_column, y_column)
            elif "聚类分布" in chart_type:
                success, fig = self.visualizer.plot_cluster_distribution()
            elif "热力图" in chart_type:
                columns = self.window.selected_columns_list.selectedItems()
                columns = [item.text() for item in columns]
                success, fig = self.visualizer.plot_heatmap(columns)
            elif "雷达图" in chart_type:
                # 准备雷达图数据

                success, fig = self.visualizer.plot_radar_chart()
            elif "健康指数" in chart_type:
                # 准备健康指数数据
                success, fig = self.visualizer.plot_health_index()
            else:
                self.show_error_message("错误", f"不支持的图表类型: {chart_type}")
                return

            if success:
                # 在界面上显示图表
                # 这里需要根据实际界面设计实现图表显示功能
                self.window.statusBar().showMessage(f"{chart_type}已刷新")
            else:
                self.show_error_message("图表生成错误", fig)

        except Exception as e:
            self.show_error_message("图表生成错误", str(e))

    def export_chart(self):
        """导出图表"""
        try:
            file_path, _ = self.window.get_save_file_path()
            if not file_path:
                return

            chart_type = self.window.chart_type_combo.currentText()

            if chart_type in self.visualizer.plots:
                fig = self.visualizer.plots[chart_type]
                fig.savefig(file_path)
                self.show_info_message("成功", f"{chart_type}已导出至: {file_path}")
            else:
                self.show_error_message("错误", f"没有找到{chart_type}的图表")

        except Exception as e:
            self.show_error_message("导出错误", str(e))

    def generate_report(self):
        """生成报告"""
        try:
            file_path, _ = self.window.get_save_file_path()
            if not file_path:
                return

            # 获取报告设置
            include_charts = self.window.include_charts_check.isChecked()
            include_tables = self.window.include_tables_check.isChecked()

            # 生成报告
            success, message = self.visualizer.generate_pdf_report(
                file_path,
                title=self.window.report_title_edit.text(),
                plots=self.visualizer.plots if include_charts else {}
            )

            if success:
                self.show_info_message("成功", message)
            else:
                self.show_error_message("报告生成错误", message)

        except Exception as e:
            self.show_error_message("报告生成错误", str(e))

    def show_info_message(self, title, message):
        """显示信息对话框"""
        QMessageBox.information(self.window, title, message, QMessageBox.Ok)

    def show_error_message(self, title, message):
        """显示错误对话框"""
        QMessageBox.critical(self.window, title, message, QMessageBox.Ok)

    def run(self):
        """运行应用"""
        self.window.show()
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    controller = AppController()
    controller.run()