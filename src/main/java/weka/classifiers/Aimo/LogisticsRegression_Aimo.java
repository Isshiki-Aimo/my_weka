package weka.classifiers.Aimo;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class LogisticsRegression_Aimo extends Classifier {
    private Instances Set_Instances;        // 实例集合
    private ReplaceMissingValues m_MissingFilter; // 数据预处理需要的过滤器
    private int item;
    private int num_class;

    private int num_instance;

    private int num_attribute;

    private int class_index;

    Matrix W;

    private double[] w;

    private double value;


    public void buildClassifier(Instances train) throws Exception {
        // 备份数据（防止更改原训练数据）
        Set_Instances = new Instances(train);

        // 先做数据预处理（填充缺失值）
        m_MissingFilter = new ReplaceMissingValues();
        m_MissingFilter.setInputFormat(Set_Instances);
        Set_Instances = Filter.useFilter(Set_Instances, m_MissingFilter);
        // 删除没有监督信息的实例
        Set_Instances.deleteWithMissingClass();


        item = 10000;
        value = 0.0003;
        num_instance = Set_Instances.numInstances();
        num_attribute = Set_Instances.numAttributes();
        class_index = Set_Instances.classIndex();
        num_class = Set_Instances.numClasses();
        w = new double[num_attribute];
        Matrix X = new Matrix(num_instance, num_attribute);
        Matrix Y = new Matrix(num_instance, 1);
        for (int i = 0; i < num_instance; i++) {
            Y.set(i, 0, Set_Instances.instance(i).classValue());
            for (int j = 0; j < num_attribute; j++) {
                if (j != class_index) {
                    X.set(i, j, Set_Instances.instance(i).value(j));
                }
                if (j == class_index) {
                    X.set(i, j, 1.0);
                }
            }
        }
        W = new Matrix(num_attribute, 1, 1);
        stocGradAscent(X, Y, item);
    }


    //计算预测值
    public double[] distributionForInstance(Instance instance) throws Exception {
        Instance transformedInstance = new Instance(instance);
        m_MissingFilter.input(transformedInstance);
        m_MissingFilter.batchFinished();
        transformedInstance = m_MissingFilter.output();
        double predicate = 0.0;
        for (int i = 0; i < num_attribute; i++) {
            if (i != class_index) {
                predicate += w[i] * transformedInstance.value(i);
            }
        }
        predicate += w[class_index];
        double[] prob = new double[num_class];
        prob[1] = sigmoid(predicate);
        prob[0] = 1 - prob[1];
        Utils.normalize(prob);//归一化
        return prob;
    }


    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }


    private void stocGradAscent(Matrix data, Matrix label, int numIter) {
        double alpha;
        for (int i = 0; i < item; i++) {
            Matrix grad;
            alpha = 2 / (1.0 + i) + 0.0001;
            Matrix error = data.times(W);
            for (int k = 0; k < num_instance; k++) {
                error.set(k, 0, sigmoid(error.get(k, 0)) - label.get(k, 0));
            }
            grad = data.transpose().times(error);
            double sum_grad = 0.0;
            for (int k = 0; k < num_attribute; k++) {
                sum_grad += grad.get(k, 0);
            }
            if (Math.abs(sum_grad) < value) {
                break;
            }
            W = W.minus(grad.times(alpha / num_instance));
        }
        w = new double[num_attribute];
        for (int i = 0; i < num_attribute; i++) {
            w[i] = W.get(i, 0);
        }
    }
}
