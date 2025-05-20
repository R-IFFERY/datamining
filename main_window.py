import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QFormLayout, QGroupBox, QLabel,
                             QLineEdit, QPushButton, QComboBox, QRadioButton, QListWidget,
                             QTableWidget, QTableWidgetItem, QSpinBox, QCheckBox, QFileDialog,
                             QMessageBox, QSplitter, QProgressBar, QDoubleSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import os


class MplCanvas(FigureCanvas):
    """Matplotlib画布，用于在PyQt5中显示图表"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("机电验证数据挖掘工具")
        self.resize(1200, 800)

        # 设置全局字体
        font = QFont("SimHei")
        self.setFont(font)

        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 菜单栏
        self.create_menu_bar()

        # 主标签页
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 添加各个功能模块标签页
        self.add_data_preprocessing_tab()
        self.add_feature_engineering_tab()
        self.add_data_mining_tab()
        self.add_model_evaluation_tab()
        self.add_visualization_tab()

        # 状态栏
        self.statusBar().showMessage("就绪")

    def create_menu_bar(self):
        menu_bar = self.menuBar()

        # 文件菜单
        file_menu = menu_bar.addMenu("文件(&F)")
        file_menu.addAction("导入数据", self.dummy_function)
        file_menu.addAction("保存配置", self.dummy_function)
        file_menu.addAction("导出报告", self.dummy_function)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close)

        # 编辑菜单
        edit_menu = menu_bar.addMenu("编辑(&E)")
        edit_menu.addAction("撤销", self.dummy_function)
        edit_menu.addAction("重做", self.dummy_function)
        edit_menu.addSeparator()
        edit_menu.addAction("复制", self.dummy_function)
        edit_menu.addAction("粘贴", self.dummy_function)

        # 视图菜单
        view_menu = menu_bar.addMenu("视图(&V)")
        view_menu.addAction("工具栏", self.dummy_function)
        view_menu.addAction("状态栏", self.dummy_function)

        # 帮助菜单
        help_menu = menu_bar.addMenu("帮助(&H)")
        help_menu.addAction("关于", self.show_about_dialog)
        help_menu.addAction("使用指南", self.dummy_function)

    def add_data_preprocessing_tab(self):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # 数据输入区
        input_group = QGroupBox("数据输入")
        input_layout = QFormLayout(input_group)

        # 数据来源选择
        source_layout = QHBoxLayout()
        self.file_radio = QRadioButton("本地文件")

        self.file_radio.setChecked(True)
        source_layout.addWidget(self.file_radio)

        source_layout.addStretch()

        input_layout.addRow("数据来源：", source_layout)

        # 文件路径选择
        path_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_file)

        path_layout.addWidget(self.file_path_edit)
        path_layout.addWidget(browse_btn)
        input_layout.addRow("文件路径：", path_layout)

        # 文件格式选择
        format_group = QGroupBox("文件格式")
        format_layout = QHBoxLayout(format_group)
        self.csv_radio = QRadioButton(".csv")
        self.excel_radio = QRadioButton(".xlsx")
        self.txt_radio = QRadioButton(".txt")
        self.csv_radio.setChecked(True)

        format_layout.addWidget(self.csv_radio)
        format_layout.addWidget(self.excel_radio)
        format_layout.addWidget(self.txt_radio)
        format_layout.addStretch()

        input_layout.addRow(format_group)

        # 预处理操作区
        operations_group = QGroupBox("预处理操作")
        operations_layout = QGridLayout(operations_group)

        # 数据清洗
        cleaning_group = QGroupBox("数据清洗")
        cleaning_layout = QVBoxLayout(cleaning_group)

        self.duplicate_check = QCheckBox("去除重复记录")
        self.missing_check = QCheckBox("处理缺失值")
        self.outlier_check = QCheckBox("检测异常值")

        cleaning_layout.addWidget(self.duplicate_check)
        cleaning_layout.addWidget(self.missing_check)
        cleaning_layout.addWidget(self.outlier_check)

        # 缺失值处理
        missing_group = QGroupBox("缺失值处理")
        missing_layout = QVBoxLayout(missing_group)

        self.missing_method_combo = QComboBox()
        self.missing_method_combo.addItems([
            "均值插补", "中位数插补", "时间序列插值"
        ])

        missing_layout.addWidget(QLabel("处理方法："))
        missing_layout.addWidget(self.missing_method_combo)

        # 异常值处理
        outlier_group = QGroupBox("异常值处理")
        outlier_layout = QVBoxLayout(outlier_group)

        self.outlier_method_combo = QComboBox()
        self.outlier_method_combo.addItems([
            "IQR方法", "Z-score", "孤立森林", "One-Class SVM"
        ])

        self.outlier_threshold_edit = QLineEdit("3.0")

        outlier_layout.addWidget(QLabel("检测方法："))
        outlier_layout.addWidget(self.outlier_method_combo)
        outlier_layout.addWidget(QLabel("阈值："))
        outlier_layout.addWidget(self.outlier_threshold_edit)

        # 标准化/归一化
        normalization_group = QGroupBox("标准化/归一化")
        normalization_layout = QVBoxLayout(normalization_group)

        self.normalization_check = QCheckBox("启用标准化/归一化")
        self.normalization_method_combo = QComboBox()
        self.normalization_method_combo.addItems([
            "Z-score标准化", "Min-Max归一化", "RobustScaler", "Normalizer"
        ])

        normalization_layout.addWidget(self.normalization_check)
        normalization_layout.addWidget(QLabel("方法："))
        normalization_layout.addWidget(self.normalization_method_combo)

        # 添加到操作区布局
        operations_layout.addWidget(cleaning_group, 0, 0)
        operations_layout.addWidget(missing_group, 0, 1)
        operations_layout.addWidget(outlier_group, 0, 2)
        operations_layout.addWidget(normalization_group, 1, 0, 1, 3)

        # 执行区
        execution_layout = QHBoxLayout()
        self.execute_preprocessing_btn = QPushButton("执行预处理")
        self.execute_preprocessing_btn.setFixedWidth(150)
        self.preview_result_btn = QPushButton("预览结果")
        self.preview_result_btn.setFixedWidth(150)
        self.save_processed_data_btn = QPushButton("保存处理后数据")
        self.save_processed_data_btn.setFixedWidth(150)

        execution_layout.addStretch()
        execution_layout.addWidget(self.execute_preprocessing_btn)
        execution_layout.addWidget(self.preview_result_btn)
        execution_layout.addWidget(self.save_processed_data_btn)
        execution_layout.addStretch()

        # 添加到标签页布局
        tab_layout.addWidget(input_group)
        tab_layout.addWidget(operations_group)
        tab_layout.addLayout(execution_layout)

        self.tab_widget.addTab(tab, "数据预处理")

    def add_feature_engineering_tab(self):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # 特征工程区域分割器
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：特征提取
        extraction_widget = QWidget()
        extraction_layout = QVBoxLayout(extraction_widget)

        # 特征类型选择
        feature_type_group = QGroupBox("特征类型")
        feature_type_layout = QVBoxLayout(feature_type_group)

        self.feature_type_radios = {}
        feature_types = [
            "时域统计特征", "频域特征", "时频特征",
            "滑窗动态特征", "物理工况特征"
        ]

        for feature_type in feature_types:
            radio = QRadioButton(feature_type)
            feature_type_layout.addWidget(radio)
            self.feature_type_radios[feature_type] = radio

        self.feature_type_radios["时域统计特征"].setChecked(True)

        # 时域特征参数
        time_domain_group = QGroupBox("时域统计特征参数")
        time_domain_layout = QFormLayout(time_domain_group)

        self.window_size_edit = QLineEdit("100")
        self.step_size_edit = QLineEdit("50")

        time_domain_layout.addRow("窗口大小：", self.window_size_edit)
        time_domain_layout.addRow("步长：", self.step_size_edit)

        # 频域特征参数
        freq_domain_group = QGroupBox("频域特征参数")
        freq_domain_layout = QFormLayout(freq_domain_group)

        self.fft_size_edit = QLineEdit("1024")
        self.freq_range_edit = QLineEdit("0-1000")

        freq_domain_layout.addRow("FFT大小：", self.fft_size_edit)
        freq_domain_layout.addRow("频率范围：", self.freq_range_edit)

        # 执行提取按钮
        self.extract_features_btn = QPushButton("执行特征提取")
        self.extract_features_btn.setFixedHeight(40)

        extraction_layout.addWidget(feature_type_group)
        extraction_layout.addWidget(time_domain_group)
        extraction_layout.addWidget(freq_domain_group)
        extraction_layout.addStretch()
        extraction_layout.addWidget(self.extract_features_btn)

        # 右侧：特征选择
        selection_widget = QWidget()
        selection_layout = QVBoxLayout(selection_widget)

        # 选择方法
        method_group = QGroupBox("特征选择方法")
        method_layout = QVBoxLayout(method_group)

        self.selection_method_checks = {}
        method_checks = [
            "方差过滤法", "互信息与相关性分析",
            "基于模型的特征重要性排序", "PCA主成分分析", "LASSO稀疏回归"
        ]

        for method in method_checks:
            check = QCheckBox(method)
            method_layout.addWidget(check)
            self.selection_method_checks[method] = check

        # 方差过滤参数
        variance_group = QGroupBox("方差过滤参数")
        variance_layout = QFormLayout(variance_group)

        self.variance_threshold_edit = QLineEdit("0.01")

        variance_layout.addRow("方差阈值：", self.variance_threshold_edit)

        # 特征列表
        feature_list_group = QGroupBox("特征列表")
        feature_list_layout = QVBoxLayout(feature_list_group)

        self.feature_table = QTableWidget(0, 4)
        self.feature_table.setHorizontalHeaderLabels(["特征名称", "类型", "重要性", "选择"])

        feature_list_layout.addWidget(self.feature_table)

        # 执行选择按钮
        self.select_features_btn = QPushButton("执行特征选择")
        self.select_features_btn.setFixedHeight(40)

        selection_layout.addWidget(method_group)
        selection_layout.addWidget(variance_group)
        selection_layout.addWidget(feature_list_group)
        selection_layout.addStretch()
        selection_layout.addWidget(self.select_features_btn)

        # 添加到分割器
        splitter.addWidget(extraction_widget)
        splitter.addWidget(selection_widget)
        splitter.setSizes([400, 600])

        # 添加到标签页布局
        tab_layout.addWidget(splitter)

        self.tab_widget.addTab(tab, "特征工程")

    def add_data_mining_tab(self):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # 算法选择区
        algorithm_group = QGroupBox("数据挖掘算法")
        algorithm_layout = QGridLayout(algorithm_group)

        # 聚类分析
        clustering_group = QGroupBox("聚类分析")
        clustering_layout = QVBoxLayout(clustering_group)

        self.clustering_algorithms = {}
        clustering_algs = ["K-Means", "DBSCAN", "谱聚类"]

        for alg in clustering_algs:
            radio = QRadioButton(alg)
            clustering_layout.addWidget(radio)
            self.clustering_algorithms[alg] = radio

        self.clustering_algorithms["K-Means"].setChecked(True)

        # 异常检测
        anomaly_group = QGroupBox("异常检测")
        anomaly_layout = QVBoxLayout(anomaly_group)

        self.anomaly_algorithms = {}
        anomaly_algs = ["孤立森林", "One-Class SVM", "局部离群因子(LOF)"]

        for alg in anomaly_algs:
            radio = QRadioButton(alg)
            anomaly_layout.addWidget(radio)
            self.anomaly_algorithms[alg] = radio

        # 预测建模
        prediction_group = QGroupBox("预测建模")
        prediction_layout = QVBoxLayout(prediction_group)

        self.prediction_algorithms = {}
        prediction_algs = ["随机森林", "XGBoost", "LSTM", "GRU", "SVR"]

        for alg in prediction_algs:
            radio = QRadioButton(alg)
            prediction_layout.addWidget(radio)
            self.prediction_algorithms[alg] = radio

        # 添加到算法选择布局
        algorithm_layout.addWidget(clustering_group, 0, 0)
        algorithm_layout.addWidget(anomaly_group, 0, 1)
        algorithm_layout.addWidget(prediction_group, 1, 0, 1, 2)

        # 参数配置区
        params_group = QGroupBox("算法参数配置")
        params_layout = QFormLayout(params_group)

        # 目标列
        self.target_column_edit = QLineEdit("label")
        params_layout.addRow("目标列：", self.target_column_edit)

        # 根据选择的算法动态显示不同的参数控件
        # 初始显示K-Means参数
        self.kmeans_param_widget = QWidget()
        kmeans_layout = QFormLayout(self.kmeans_param_widget)

        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(2, 20)
        self.n_clusters_spin.setValue(5)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(50, 1000)
        self.max_iter_spin.setValue(300)

        kmeans_layout.addRow("聚类数(K)：", self.n_clusters_spin)
        kmeans_layout.addRow("最大迭代次数：", self.max_iter_spin)

        # 异常检测参数
        self.anomaly_param_widget = QWidget()
        anomaly_layout = QFormLayout(self.anomaly_param_widget)

        self.contamination_edit = QLineEdit("0.05")

        anomaly_layout.addRow("异常比例：", self.contamination_edit)

        # 预测模型参数
        self.prediction_param_widget = QWidget()
        prediction_layout = QFormLayout(self.prediction_param_widget)

        self.test_size_edit = QLineEdit("0.2")
        self.n_estimators_edit = QLineEdit("100")

        prediction_layout.addRow("测试集比例：", self.test_size_edit)
        prediction_layout.addRow("树的数量：", self.n_estimators_edit)

        # 添加到参数布局
        params_layout.addRow("算法参数：", self.kmeans_param_widget)

        # 执行区
        execution_layout = QHBoxLayout()
        self.run_algorithm_btn = QPushButton("运行算法")
        self.run_algorithm_btn.setFixedWidth(150)
        self.save_model_btn = QPushButton("保存模型")
        self.save_model_btn.setFixedWidth(120)

        execution_layout.addStretch()
        execution_layout.addWidget(self.run_algorithm_btn)
        execution_layout.addWidget(self.save_model_btn)
        execution_layout.addStretch()

        # 添加到标签页布局
        tab_layout.addWidget(algorithm_group)
        tab_layout.addWidget(params_group)
        tab_layout.addLayout(execution_layout)

        self.tab_widget.addTab(tab, "数据挖掘算法")

    def add_model_evaluation_tab(self):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # 评估指标区
        metrics_group = QGroupBox("模型评估指标")
        metrics_layout = QHBoxLayout(metrics_group)

        # 分类模型评估
        classification_group = QGroupBox("分类模型评估")
        classification_layout = QGridLayout(classification_group)

        classification_metrics = [
            "准确率 (Accuracy)", "精确率 (Precision)",
            "召回率 (Recall)", "F1分数 (F1-Score)",
            "AUC值", "Kappa系数"
        ]

        self.classification_metric_labels = {}
        for i, metric in enumerate(classification_metrics):
            classification_layout.addWidget(QLabel(metric), i, 0)
            value_label = QLabel("0.00")
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setStyleSheet("font-weight: bold; color: #0066CC;")
            classification_layout.addWidget(value_label, i, 1)
            self.classification_metric_labels[metric] = value_label

        # 回归模型评估
        regression_group = QGroupBox("回归模型评估")
        regression_layout = QGridLayout(regression_group)

        regression_metrics = [
            "均方误差 (MSE)", "均方根误差 (RMSE)",
            "平均绝对误差 (MAE)", "R²分数 (R²-Score)",
            "解释方差分数", "中位数绝对误差"
        ]

        self.regression_metric_labels = {}
        for i, metric in enumerate(regression_metrics):
            regression_layout.addWidget(QLabel(metric), i, 0)
            value_label = QLabel("0.00")
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setStyleSheet("font-weight: bold; color: #0066CC;")
            regression_layout.addWidget(value_label, i, 1)
            self.regression_metric_labels[metric] = value_label

        metrics_layout.addWidget(classification_group)
        metrics_layout.addWidget(regression_group)

        # 评估方法区
        method_group = QGroupBox("评估方法")
        method_layout = QHBoxLayout(method_group)

        self.cross_val_radio = QRadioButton("交叉验证")
        self.train_test_radio = QRadioButton("训练集/测试集分割")
        self.train_test_radio.setChecked(True)

        self.split_ratio_edit = QLineEdit("0.8")

        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(2, 10)
        self.cv_folds_spin.setValue(5)

        method_layout.addWidget(self.cross_val_radio)
        method_layout.addWidget(QLabel("折数："))
        method_layout.addWidget(self.cv_folds_spin)
        method_layout.addWidget(self.train_test_radio)
        method_layout.addWidget(QLabel("分割比例："))
        method_layout.addWidget(self.split_ratio_edit)
        method_layout.addStretch()

        # 执行区
        execution_layout = QHBoxLayout()
        self.evaluate_btn = QPushButton("执行评估")
        self.evaluate_btn.setFixedWidth(150)
        self.compare_btn = QPushButton("模型比较")
        self.compare_btn.setFixedWidth(150)
        self.export_report_btn = QPushButton("导出评估报告")
        self.export_report_btn.setFixedWidth(150)

        execution_layout.addStretch()
        execution_layout.addWidget(self.evaluate_btn)
        execution_layout.addWidget(self.compare_btn)
        execution_layout.addWidget(self.export_report_btn)
        execution_layout.addStretch()

        # 添加到标签页布局
        tab_layout.addWidget(metrics_group)
        tab_layout.addWidget(method_group)
        tab_layout.addLayout(execution_layout)

        self.tab_widget.addTab(tab, "模型评估与优化")

    def add_visualization_tab(self):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # 图表类型选择区
        chart_type_group = QGroupBox("图表类型")
        chart_type_layout = QHBoxLayout(chart_type_group)

        chart_types = [
            "趋势图", "聚类分布图", "热力图", "雷达图",
            "健康指数曲线", "高维投影图", "混淆矩阵"
        ]

        self.chart_type_buttons = {}
        for chart_type in chart_types:
            btn = QPushButton(chart_type)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, ct=chart_type: self.on_chart_type_selected(ct))
            chart_type_layout.addWidget(btn)
            self.chart_type_buttons[chart_type] = btn

        self.chart_type_buttons["趋势图"].setChecked(True)

        # 图表参数设置区
        chart_param_group = QGroupBox("图表参数")
        chart_param_layout = QFormLayout(chart_param_group)

        self.x_column_combo = QComboBox()
        self.y_column_combo = QComboBox()

        chart_param_layout.addRow("X轴列：", self.x_column_combo)
        chart_param_layout.addRow("Y轴列：", self.y_column_combo)

        # 图表展示区
        chart_display_group = QGroupBox("图表展示")
        chart_display_layout = QVBoxLayout(chart_display_group)

        # 创建Matplotlib画布
        self.chart_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        chart_display_layout.addWidget(self.chart_canvas)

        # 图表控制区
        chart_control_layout = QHBoxLayout()

        self.refresh_chart_btn = QPushButton("刷新图表")
        self.export_chart_btn = QPushButton("导出图表")
        self.generate_report_btn = QPushButton("生成报告")

        chart_control_layout.addWidget(self.refresh_chart_btn)
        chart_control_layout.addWidget(self.export_chart_btn)
        chart_control_layout.addWidget(self.generate_report_btn)
        chart_control_layout.addStretch()

        chart_display_layout.addLayout(chart_control_layout)

        # 报告生成区
        report_group = QGroupBox("自动报告生成")
        report_layout = QFormLayout(report_group)

        self.report_title_edit = QLineEdit("机电系统数据挖掘分析报告")

        self.include_charts_check = QCheckBox("包含所有图表")
        self.include_charts_check.setChecked(True)

        self.include_tables_check = QCheckBox("包含数据表格")
        self.include_tables_check.setChecked(True)

        report_layout.addRow("报告标题：", self.report_title_edit)
        report_layout.addRow(self.include_charts_check)
        report_layout.addRow(self.include_tables_check)

        # 添加到标签页布局
        tab_layout.addWidget(chart_type_group)
        tab_layout.addWidget(chart_param_group)
        tab_layout.addWidget(chart_display_group)
        tab_layout.addWidget(report_group)

        self.tab_widget.addTab(tab, "可视化展示与报告")

    def on_chart_type_selected(self, chart_type):
        """图表类型选择事件处理"""
        for ct, btn in self.chart_type_buttons.items():
            btn.setChecked(ct == chart_type)

    def browse_file(self):
        """浏览文件对话框"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "", "数据文件 (*.csv *.xlsx *.txt);;所有文件 (*)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)
            # 根据文件扩展名自动选择格式
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                self.csv_radio.setChecked(True)
            elif ext == '.xlsx':
                self.excel_radio.setChecked(True)
            elif ext == '.txt':
                self.txt_radio.setChecked(True)

    def get_save_file_path(self):
        """获取保存文件路径"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存文件", "", "所有文件 (*.*)"
        )
        return file_path, _

    def show_about_dialog(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于",
                          "机电验证数据挖掘工具\n\n"
                          "版本: 1.0.0\n"
                          "用于机电设备验证数据的分析与挖掘\n"
                          "支持数据预处理、特征工程、数据挖掘和可视化分析"
                          )

    def dummy_function(self):
        """占位函数，实际实现中需要替换为具体功能"""
        self.statusBar().showMessage("功能开发中...")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用Fusion风格，跨平台一致性更好

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())