package weka.classifiers.Aimo;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.util.Random;

public class BP_Aimo extends Classifier {
    private Instances Set_Instances;        // 实例集合
    private ReplaceMissingValues m_MissingFilter; // 数据预处理需要的过滤器

    private int Num_Attributes;              // 属性个数
    private int Num_Instances;              // 实例个数
    private int Num_Classes;                // 类别个数

    private int hidden_size = 12;                // 隐藏层节点个数


    private Matrix input;
    private Matrix hidden;
    private Matrix output;
    private Matrix target;

    private Matrix hidDelta;
    private Matrix outDelta;

    private double eta = 0.25;                // 学习率
    private double omiga = 0.9;

    private Matrix iptHidWeights;
    private Matrix hidOptWeights;

    private Matrix iptHidPrevUptWeights;
    private Matrix hidOptPrevUptWeights;

    public double optErrSum = 0d;
    public double hidErrSum = 0d;

    private final Random random = new Random(777777);


    @Override
    public void buildClassifier(Instances train) throws Exception {
        // 备份数据（防止更改原训练数据）
        Set_Instances = new Instances(train);

        // 先做数据预处理（填充缺失值）
        m_MissingFilter = new ReplaceMissingValues();
        m_MissingFilter.setInputFormat(Set_Instances);
        Set_Instances = Filter.useFilter(Set_Instances, m_MissingFilter);
        // 删除没有监督信息的实例
        Set_Instances.deleteWithMissingClass();

        Num_Attributes = Set_Instances.numAttributes();
        Num_Instances = Set_Instances.numInstances();
        Num_Classes = Set_Instances.numClasses();

        // 初始化输入层、隐藏层、输出层、目标层
        input = new Matrix(Num_Attributes + 1, 1);
        hidden = new Matrix(hidden_size + 1, 1);
        output = new Matrix(Num_Classes + 1, 1);
        target = new Matrix(Num_Classes + 1, 1);

        // 初始化隐藏层的delta
        hidDelta = new Matrix(hidden_size + 1, 1);
        // 初始化输出层的delta
        outDelta = new Matrix(Num_Classes + 1, 1);

        // 初始化权重矩阵
        iptHidWeights = new Matrix(Num_Attributes + 1, hidden_size + 1);
        hidOptWeights = new Matrix(hidden_size + 1, Num_Classes + 1);
        randomizeWeights(iptHidWeights);
        randomizeWeights(hidOptWeights);

        // 初始化权重更新矩阵
        iptHidPrevUptWeights = new Matrix(Num_Attributes + 1, hidden_size + 1);
        hidOptPrevUptWeights = new Matrix(hidden_size + 1, Num_Classes + 1);


    }

    private void randomizeWeights(Matrix weights) {
        for (int i = 0; i < weights.getRowDimension(); i++) {
            for (int j = 0; j < weights.getColumnDimension(); j++) {
                double randomValue = random.nextDouble();
                weights.set(i, j, random.nextDouble() > 0.5 ? randomValue : -randomValue);
            }
        }
    }




}
