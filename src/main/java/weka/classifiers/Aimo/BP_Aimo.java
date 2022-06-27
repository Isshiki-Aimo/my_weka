package weka.classifiers.Aimo;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

public class BP_Aimo extends Classifier {
    private Instances Set_Instances;        // 实例集合
    private ReplaceMissingValues m_MissingFilter; // 数据预处理需要的过滤器

    private int Num_Attributes;              // 属性个数
    private int Num_Instances;              // 实例个数
    private int Num_Classes;                // 类别个数

    private int hidden_size = 6;                // 隐藏层节点个数


    private Matrix input;
    private Matrix hidden;
    private Matrix output;
    private Matrix target;

    private Matrix hidDelta;
    private Matrix outDelta;

    private double eta = 0.2;                // 学习率
    private double omiga = 0.3;
    private int iteration = 600;            // 迭代次数

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

        Num_Attributes = Set_Instances.numAttributes() - 1;
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

        //开始训练
        for (int iter = 0; iter <= iteration; iter++) {
            for (int i = 0; i < Num_Instances; i++) {
                loadtarget(target, Set_Instances.instance(i));
                loaddata(input, Set_Instances.instance(i));
                train(input, target);
            }
        }
    }

    public double[] distributionForInstance(Instance instance) {
        // 备份数据
        Instance transformedInstance = new Instance(instance);

        // 先做预处理
        m_MissingFilter.input(transformedInstance);
        m_MissingFilter.batchFinished();
        transformedInstance = m_MissingFilter.output();

        loaddata(input, transformedInstance);
        forward();
        return getNetworkOutput();
    }

    private void randomizeWeights(Matrix weights) {
        //矩阵扩张一个维度
        for (int i = 0; i < weights.getRowDimension(); i++) {
            if (i == 0) {
                weights.set(i, 0, 1);
            } else {
                weights.set(i, 0, 0);
            }
        }

        //随机初始化权重
        for (int i = 0; i < weights.getRowDimension(); i++) {
            for (int j = 1; j < weights.getColumnDimension(); j++) {
                double randomValue = random.nextDouble();
                weights.set(i, j, random.nextDouble() > 0.5 ? randomValue : -randomValue);
            }
        }
    }

    private double sigmoid(double val) {
        return 1d / (1d + Math.exp(-val));
    }

    private void forward(Matrix layer0, Matrix layer1, Matrix weight) {
        layer0.set(0, 0, 1d);
        layer1.set(0, 0, 1d);
        Matrix temp = layer0.transpose().times(weight);
        temp = temp.transpose();
        for (int i = 1; i < temp.getRowDimension(); i++) {
            layer1.set(i, 0, sigmoid(temp.get(i, 0)));
        }
    }

    private void forward() {
        forward(input, hidden, iptHidWeights);
        forward(hidden, output, hidOptWeights);
    }

    private void outputErr() {
        double errSum = 0;
        for (int idx = 1, len = outDelta.getRowDimension(); idx != len; ++idx) {
            double o = output.get(idx, 0);
            outDelta.set(idx, 0, o * (1 - o) * (target.get(idx, 0) - o));
            errSum += Math.abs(outDelta.get(idx, 0));
        }
        optErrSum = errSum;
    }

    private void hiddenErr() {
        double errSum = 0;
        for (int idx = 1, len = hidDelta.getRowDimension(); idx != len; ++idx) {
            double o = hidden.get(idx, 0);
            double sum = 0;
            for (int jdx = 1, len2 = outDelta.getRowDimension(); jdx != len2; ++jdx) {
                sum += outDelta.get(jdx, 0) * hidOptWeights.get(idx, jdx);
                hidDelta.set(idx, 0, o * (1d - o) * sum);
                errSum += Math.abs(hidDelta.get(idx, 0));
            }
        }
        hidErrSum = errSum;
    }

    private void calculateDelta() {
        outputErr();
        hiddenErr();
    }

    private void updateWeights(Matrix delta, Matrix layer, Matrix weights, Matrix prevWeights) {
        layer.set(0, 0, 1d);
        for (int i = 1, len = delta.getRowDimension(); i != len; ++i) {
            for (int j = 0, len2 = layer.getRowDimension(); j != len2; ++j) {
                double newWeight = omiga * prevWeights.get(j, i) + eta * delta.get(i, 0) * layer.get(j, 0);
                weights.set(j, i, weights.get(j, i) + newWeight);
                prevWeights.set(j, i, newWeight);
            }
        }
    }

    private void updateWeights() {
        updateWeights(hidDelta, input, iptHidWeights, iptHidPrevUptWeights);
        updateWeights(outDelta, hidden, hidOptWeights, hidOptPrevUptWeights);
    }

    private void train(Matrix input, Matrix target) {
        this.input = input;
        this.target = target;
        forward();
        calculateDelta();
        updateWeights();
    }

    private void loaddata(Matrix A, Instance data) {
        for (int i = 0; i < Num_Attributes; i++) {
            A.set(i + 1, 0, data.value(i));
        }
    }

    private void loadtarget(Matrix A, Instance data) {
        for (int i = 0; i < Num_Classes; i++) {
            if (data.classValue() == i)
                A.set(i + 1, 0, 1);
            else
                A.set(i + 1, 0, 0);
        }
    }

    private int find_max_idx(double[] arr) {
        int max_idx = 0;
        double max_val = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max_val) {
                max_idx = i;
                max_val = arr[i];
            }
        }
        return max_idx;
    }

    private double[] getNetworkOutput() {
        double[] result = new double[Num_Classes];
        for (int i = 1; i != Num_Classes + 1; i++) {
            result[i - 1] = output.get(i, 0);
        }

        int max_idx = find_max_idx(result);
        for (int i = 0; i < result.length; i++) {
            if (i == max_idx) {
                result[i] = 1;
            } else {
                result[i] = 0;
            }
        }
        return result;
    }


    public static void main(String[] args) {
        try {
            String path = "C:\\Users\\Aimo\\Documents\\My_test_data\\Weak_data\\iris.arff";
            Instances linear = new Instances(new BufferedReader(new FileReader(path)));
            linear.setClassIndex(linear.numAttributes() - 1);
            int num_attributes = linear.numAttributes();
            System.out.println("num_attributes: " + num_attributes);
            System.out.println("class");
            System.out.println(linear.instance(0).classValue());

            System.out.println("index value");
            for (int i = 0; i < num_attributes; i++) {
                System.out.println(linear.instance(0).value(i));
            }

            BP_Aimo bp = new BP_Aimo();
            bp.buildClassifier(linear);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}

