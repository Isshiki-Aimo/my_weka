package weka.classifiers.Aimo;


import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class LinearRegression_Aimo extends Classifier{

    // 数据成员
    private double[] Wb;				// 参数数组
    private Instances Set_Instances;		// 实例集合
    private int Num_Attributes;				// 属性个数
    private int Num_Instances;			// 实例个数

    private ReplaceMissingValues m_MissingFilter; // 数据预处理需要的过滤器

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
        Matrix Matrix_Y = new Matrix(Num_Instances,1);
        for(int i = 0; i< Num_Instances; i++) {
            Matrix_Y.set(i, 0, Set_Instances.instance(i).classValue());
            for(int j = 0; j< Num_Attributes -1; j++) {
                Matrix_X.set(i, j, Set_Instances.instance(i).value(j));
            }
            Matrix_X.set(i, Num_Attributes -1, 1);
        }
        // 最小二乘法求解
        boolean success = true;
        double LambdaI = 0.1;
        Matrix solution = new Matrix(Num_Attributes, 1);
        do {
            Matrix X_Xt = Matrix_X.transpose().times(Matrix_X);
            // 对角线加上一个值保证满秩
            for (int i = 0; i < Num_Attributes; i++)
                X_Xt.set(i, i, X_Xt.get(i, i) + LambdaI);
            Matrix X_Y = Matrix_X.transpose().times(Matrix_Y);
            try {
                solution = X_Xt.solve(X_Y);
                success = true;
            }
            catch (Exception ex) {
                LambdaI *= 10;
                success = false;
            }
        } while (!success);
        for(int i = 0; i< Num_Attributes; i++) {
            Wb[i] = solution.get(i, 0);
        }
    }

    // 预测函数
    public double classifyInstance(Instance instance) throws Exception {
        // 备份数据
        Instance transformedInstance = new Instance(instance);

        // 先做预处理
        m_MissingFilter.input(transformedInstance);
        m_MissingFilter.batchFinished();
        transformedInstance = m_MissingFilter.output();

        // 计算最终结果
        double temp = 0;
        for(int i = 0; i< Num_Attributes -1; i++) {
            temp += Wb[i] * transformedInstance.value(i);
        }
        temp += Wb[Num_Attributes -1];
        return temp;
    }

    public static void main(String[] args) {
        try {
            String path = "C:\\Users\\Aimo\\Documents\\My_test_data\\Regression36\\fruitfly.arff";
            Instances linear = new Instances(new BufferedReader(new FileReader(path)));
            int num_attributes = linear.numAttributes();
            linear.setClassIndex(linear.numAttributes() - 1);
            LinearRegression_Aimo linearRegression = new LinearRegression_Aimo();
            linearRegression.buildClassifier(linear);
            System.out.println("class");
            System.out.println(linear.instance(0).classValue());
            System.out.println("wb");
            for (int i = 0; i < num_attributes; i++) {
                System.out.println(linearRegression.Wb[i]);
            }
            System.out.println("index value");
            for (int i = 0; i < num_attributes; i++) {
                System.out.println(linear.instance(0).value(i));
            }
            System.out.println("predict");
            System.out.println(linearRegression.classifyInstance(linear.instance(0)));
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
