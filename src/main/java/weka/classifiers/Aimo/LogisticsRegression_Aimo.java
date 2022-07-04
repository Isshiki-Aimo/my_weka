package weka.classifiers.Aimo;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.matrix.Matrix;

public class LogisticsRegression_Aimo extends Classifier {
    private int item;
    private int num_class;

    private int num_instance;

    private int num_attribute;

    private int class_index;

    Matrix W;

    private double[] w;

    private double value;


    public void buildClassifier(Instances train) throws Exception {
        item = 10000;
        value = 0.0003;
        num_instance = train.numInstances();
        num_attribute = train.numAttributes();
        class_index = train.classIndex();
        num_class = train.numClasses();
        w = new double[num_attribute];
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
        W = new Matrix(num_attribute, 1, 1);
        stocGradAscent(X, Y, item);
    }


    //计算预测值
    public double[] distributionForInstance(Instance instance) throws Exception {
        double predicate = 0.0;
        for (int i = 0; i < num_attribute; i++) {
            if (i != class_index) {
                predicate += w[i] * instance.value(i);
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
