package weka.classifiers.Aimo;

import weka.core.*;
import weka.classifiers.*;

import java.io.BufferedReader;
import java.io.FileReader;

public class NaiveBayes_Aimo extends Classifier {

    //数据集中的类数量
    private int Num_Class;

    //数据集中包括在类内的属性数量
    private int Num_Attributes;

    //数据集中的实例数
    private int Num_Instances;

    //  数据集中出现的类数和每个属性值出现的次数
    private double[][] Att_in_each_class;

    //数据集中出现的每个类的数量
    private double[] Class_Count;

    //数据集中每个属性的种类
    private int[] Num_Att_Values;

    //数据集中每个属性的起始索引
    private int[] m_StartAttIndex;

    //数据集中所有属性的种类
    private int m_TotalAttValues;

    //数据集中类属性的索引
    private int ClassIndex;


    public void buildClassifier(Instances instances) throws Exception {
        Num_Class = instances.numClasses();
        ClassIndex = instances.classIndex();
        Num_Attributes = instances.numAttributes();
        Num_Instances = instances.numInstances();

        m_TotalAttValues = 0;
        m_StartAttIndex = new int[Num_Attributes];
        Num_Att_Values = new int[Num_Attributes];
        for (int i = 0; i < Num_Attributes; i++) {
            if (i != ClassIndex) {
                //获取每个属性的起始位置
                m_StartAttIndex[i] = m_TotalAttValues;
                //获取每个属性的种类数
                Num_Att_Values[i] = instances.attribute(i).numValues();
                m_TotalAttValues += Num_Att_Values[i];
            } else {
                m_StartAttIndex[i] = -1;
                Num_Att_Values[i] = Num_Class;
            }
        }
        Class_Count = new double[Num_Class];
        Att_in_each_class = new double[Num_Class][m_TotalAttValues];
        for (int k = 0; k < Num_Instances; k++) {
            //获取数据集中各类别的数量
            int classVal = (int) instances.instance(k).classValue();
            Class_Count[classVal]++;
            int[] attIndex = new int[Num_Attributes];
            for (int i = 0; i < Num_Attributes; i++) {
                if (i == ClassIndex) {
                    attIndex[i] = -1;
                } else {
                    //获取数据集中每个属性的数量
                    attIndex[i] = m_StartAttIndex[i] + (int) instances.instance(k).value(i);
                    Att_in_each_class[classVal][attIndex[i]]++;
                }
            }
        }
    }


    public double[] distributionForInstance(Instance instance) throws Exception {

        double[] probs = new double[Num_Class];

        int[] attIndex = new int[Num_Attributes];
        for (int att = 0; att < Num_Attributes; att++) {
            if (att == ClassIndex)
                attIndex[att] = -1;
            else
                attIndex[att] = m_StartAttIndex[att] + (int) instance.value(att);
        }

        for (int classVal = 0; classVal < Num_Class; classVal++) {
            probs[classVal] = (Class_Count[classVal] + 1.0) / (Num_Instances + Num_Class);
            for (int att = 0; att < Num_Attributes; att++) {
                if (attIndex[att] == -1) continue;
                probs[classVal] *= (Att_in_each_class[classVal][attIndex[att]] + 1.0) / (Class_Count[classVal] + Num_Att_Values[att]);
            }
        }

        Utils.normalize(probs);
        return probs;
    }

    public static void main(String[] args) {
        try {
            String path = "C:\\Users\\Aimo\\Documents\\My_test_data\\Weak_data\\vote.arff";
            Instances linear = new Instances(new BufferedReader(new FileReader(path)));
            linear.setClassIndex(linear.numAttributes() - 1);
            int numClasses = linear.numClasses();
            int classIndex = linear.classIndex();
            int numAttributes = linear.numAttributes();
            int numInstances = linear.numInstances();
            System.out.println("numClasses: " + numClasses);
            System.out.println("classIndex: " + classIndex);
            System.out.println("numAttributes: " + numAttributes);
            System.out.println("numInstances: " + numInstances);
            for (int i = 0; i < numAttributes; i++) {
                System.out.println(linear.instance(0).value(i));
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}