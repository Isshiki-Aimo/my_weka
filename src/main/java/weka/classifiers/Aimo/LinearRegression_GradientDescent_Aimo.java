package weka.classifiers.Aimo;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.BufferedReader;
import java.io.FileReader;


public class LinearRegression_GradientDescent_Aimo extends Classifier {
    // 数据成员
    private double[] Wb;                // 参数数组
    private Instances Set_Instances;        // 实例集合
    private int Num_Attributes;                // 属性个数
    private int Num_Instances;            // 实例个数

    private ReplaceMissingValues m_MissingFilter; // 数据预处理需要的过滤器

    private Matrix Compute_Gradient(final Matrix x, final Matrix y, final Matrix theta, int Batch_size) {
        Matrix gradient = new Matrix(Num_Attributes, 1);
        Matrix h_theta;

        h_theta = x.times(theta);

        for (int i = 0; i < Num_Attributes; i++) {
            //求一个属性的梯度
            double sum = 0;
            for (int j = 0; j < Batch_size; j++) {
                sum += ((h_theta.get(j, 0) - y.get(j, 0)) * x.get(j, i));
            }
            double gradient_i = sum / Batch_size;
            gradient.set(i, 0, gradient_i);
        }
        return gradient;
    }

    private Matrix Update_thetas(Matrix thetas, Matrix gradient, double stepSize) {
        Matrix new_thetas = new Matrix(Num_Attributes, 1);
        for (int i = 0; i < Num_Attributes; i++) {
            new_thetas.set(i, 0, thetas.get(i, 0) - stepSize * gradient.get(i, 0));
        }
        return new_thetas;
    }

    private double judge_stop(Matrix gradient_new, Matrix gradient_old) {
        double distance = 0;
        for (int i = 0; i < Num_Attributes; i++) {
            distance += Math.pow(gradient_new.get(i, 0) - gradient_old.get(i, 0), 2);
        }
        return Math.sqrt(distance);
    }

    private Matrix Gradient_Descent(double stepSize, final Matrix x, final Matrix y, double tolerance, int maxIter) {
        Matrix thetas = new Matrix(Num_Attributes, 1);
        //第一轮参数全部为0
        for (int i = 0; i < Num_Attributes; i++) {
            thetas.set(i, 0, 0.0);
        }
        Matrix gradient_old = new Matrix(Num_Attributes, 1);
        Matrix gradient_new = new Matrix(Num_Attributes, 1);
        for (int i = 0; i < Num_Attributes; i++) {
            gradient_old.set(i, 0, 0.0);
            gradient_new.set(i, 0, 0.0);
        }
        int iter = 0;
        while (iter < maxIter) {
            gradient_old = gradient_new;
            gradient_new = Compute_Gradient(x, y, thetas, x.getRowDimension());
            double distance = judge_stop(gradient_new, gradient_old);
            if (distance < tolerance) {
                break;
            }
            thetas = Update_thetas(thetas, gradient_new, stepSize);
            iter++;
        }
        return thetas;
    }


    /**
     * @param train 训练数据
     * @throws Exception 异常
     */
    // 训练函数
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
        Wb = new double[Num_Attributes];

        // 矩阵初始化
        Matrix Matrix_X = new Matrix(Num_Instances, Num_Attributes);
        Matrix Matrix_Y = new Matrix(Num_Instances, 1);
        for (int i = 0; i < Num_Instances; i++) {
            Matrix_Y.set(i, 0, Set_Instances.instance(i).classValue());
            for (int j = 0; j < Num_Attributes - 1; j++) {
                Matrix_X.set(i, j, Set_Instances.instance(i).value(j));
            }
            Matrix_X.set(i, Num_Attributes - 1, 1);
        }
        // 求解最优参数

        Matrix thetas = Gradient_Descent(0.0001, Matrix_X, Matrix_Y, 0.0000001, 10000000);

        // 将参数赋值给Wb
        for (int i = 0; i < Num_Attributes; i++) {
            Wb[i] = thetas.get(i, 0);
        }
    }

    /**
     * @param instance 待预测实例
     * @return 预测结果
     * @throws Exception 异常
     */
    public double classifyInstance(Instance instance) throws Exception {
        // 备份数据
        Instance transformedInstance = new Instance(instance);

        // 先做预处理
        m_MissingFilter.input(transformedInstance);
        m_MissingFilter.batchFinished();
        transformedInstance = m_MissingFilter.output();

        // 计算最终结果
        double temp = 0;
        for (int i = 0; i < Num_Attributes - 1; i++) {
            temp += Wb[i] * transformedInstance.value(i);
        }
        temp += Wb[Num_Attributes - 1];
        return temp;
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


//            LinearRegression_GradientDescent_Aimo linearRegression = new LinearRegression_GradientDescent_Aimo();
//            linearRegression.buildClassifier(linear);
//
//            System.out.println("wb");
//            for (int i = 0; i < num_attributes; i++) {
//                System.out.println(linearRegression.Wb[i]);
//            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
