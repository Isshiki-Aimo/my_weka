package weka.classifiers.Aimo;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.matrix.Matrix;

public class LogisticsRegression_Aimo extends Classifier {
    //设置最大迭代次数
    private int item;
    //
    private int num_class;

    //训练样本个数
    private int num_instance;

    //属性数量
    private int num_attribute;

    private int class_index;

    //存储回归变量
    private double[] w;

    //设置梯度值的阈值
    private double value;


    public void buildClassifier(Instances train) throws Exception {
        item = 10000;
        value = 0.0003;
        num_instance = train.numInstances();
        num_attribute = train.numAttributes();
        class_index = train.classIndex();
        num_class = train.numClasses();
        w = new double[num_attribute];//一类属性作为预测变量，但增加一个常数项
        //构造矩阵
        Matrix X = new Matrix(num_instance, num_attribute);
        Matrix Y = new Matrix(num_instance, 1);
        for (int i = 0; i < num_instance; i++) {
            Y.set(i, 0, train.instance(i).classValue());
            for (int j = 0; j < num_attribute; j++) {
                if (j != class_index) {
                    X.set(i, j, train.instance(i).value(j));
                }
                if (j == class_index) {
                    X.set(i, j, 1.0);
                }
            }
        }
        Matrix W = new Matrix(num_attribute, 1, 1);
        double alpha = 0.0;
        for (int i = 0; i < item; i++) {
            Matrix temp = new Matrix(num_attribute, 1);
            alpha = 4 / (1.0 + i) + 0.0001;//动态调整学习率，刚开始学习率大，随着迭代次数的增加，学习率逐渐衰减
            Matrix sum = X.times(W);//X（num_instance,num_attribute） W(num_attribute,1),sum(num_instance,1)
            for (int k = 0; k < num_instance; k++) {
                sum.set(k, 0, (1.0 / (1.0 + Math.exp(-sum.get(k, 0)))) - Y.get(k, 0));//求解h（xi）-yi,(n_instance,1)
            }
            temp = X.transpose().times(sum);//(n_attribute,1)
            double sum_grad = 0.0;
            for (int k = 0; k < num_attribute; k++) {
                sum_grad += temp.get(k, 0);
            }
            if (sum_grad < value) {
                break;
            }
            W = W.minus(temp.times(alpha / num_instance));//更新后的W
        }
        w = new double[num_attribute];
        for (int i = 0; i < num_attribute; i++) {
            w[i] = W.get(i, 0);
        }
    }


    //计算预测值
    public double[] distributionForInstance(Instance instance) throws Exception {
        double temp = 0.0;
        for (int i = 0; i < num_attribute; i++) {
            if (i != class_index) {
                temp += w[i] * instance.value(i);
            }
        }
        temp += w[class_index];
        double[] prob = new double[num_class];
//        prob[1]=1/(1+Math.exp(-temp));
//        prob[0]=1-prob[1];
        prob[1] = Math.exp(temp) / (1 + Math.exp(temp));
        prob[0] = 1 - prob[1];
        Utils.normalize(prob);//归一化
        return prob;
    }

}
