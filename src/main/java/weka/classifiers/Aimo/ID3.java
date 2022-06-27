package weka.classifiers.Aimo;

import java.util.Arrays;
import java.util.Enumeration;

import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.Capabilities.Capability;

public class ID3 extends Classifier {

    //子节点
    private ID3[] children;

    //用于划分的属性
    private Attribute splitAttribute;


    //叶子节点的类分布
    private double[] classDistributions;

    // 类属性
    private Attribute classAttribute;

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }


    public void buildClassifier(Instances data) throws Exception {

        // 检测是否可以创建分类器
        getCapabilities().testWithFail(data);

        // 删除缺少类的实例
        data = new Instances(data);
        data.deleteWithMissingClass();

        makeTree(data);
    }

    private void makeTree(Instances data) throws Exception {

        // 检测节点中是否有实例
        if (data.numInstances() == 0) {
            splitAttribute = null;
            classDistributions = new double[data.numClasses()];
        } else {
            // 寻找最大infoGains的属性
            double[] infoGains = new double[data.numAttributes()];
            Enumeration attEnum = data.enumerateAttributes();
            while (attEnum.hasMoreElements()) {
                Attribute att = (Attribute) attEnum.nextElement();
                infoGains[att.index()] = computeInfoGain(data, att);
            }

            // 检查最大infoGains的属性是否为空
            int maxIG = maxIndex(infoGains);
            if (maxIG != -1) {
                splitAttribute = data.attribute(maxIndex(infoGains));
            } else {
                throw new Exception("array null");
            }

            // 如果infoGains为0，则为叶节点
            if (Double.compare(infoGains[splitAttribute.index()], 0) == 0) {
                splitAttribute = null;

                classDistributions = new double[data.numClasses()];
                for (int i = 0; i < data.numInstances(); i++) {
                    Instance inst = (Instance) data.instance(i);
                    classDistributions[(int) inst.classValue()]++;
                }
                Utils.normalize(classDistributions);
                classAttribute = data.classAttribute();
            } else {
                //在此节点下创建树
                Instances[] splitData = splitData(data, splitAttribute);
                children = new ID3[splitAttribute.numValues()];
                for (int j = 0; j < splitAttribute.numValues(); j++) {
                    children[j] = new ID3();
                    children[j].makeTree(splitData[j]);
                }
            }
        }
    }


    private static int maxIndex(double[] array) {
        int max = 0;

        if (array.length > 0) {
            for (int i = 1; i < array.length; ++i) {
                if (array[i] > array[max]) {
                    max = i;
                }
            }
            return max;
        } else {
            return -1;
        }
    }


    public double[] distributionForInstance(Instance instance)
            throws NoSupportForMissingValuesException {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("NewID3: Cannot handle missing values");
        }
        if (splitAttribute == null) {
            System.out.println(Arrays.toString(classDistributions));
            return classDistributions;
        } else {
            return children[(int) instance.value(splitAttribute)].
                    distributionForInstance(instance);
        }
    }


    private double computeInfoGain(Instances data, Attribute att) {

        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (Instances splitdata : splitData) {
            if (splitdata.numInstances() > 0) {
                double splitNumInstances = splitdata.numInstances();
                double dataNumInstances = data.numInstances();
                double proportion = splitNumInstances / dataNumInstances;
                infoGain -= proportion * computeEntropy(splitdata);
            }
        }
        return infoGain;
    }


    private static double computeEntropy(Instances data) {

        double[] labelCounts = new double[data.numClasses()];
        for (int i = 0; i < data.numInstances(); ++i) {
            labelCounts[(int) data.instance(i).classValue()]++;
        }

        double entropy = 0;
        for (double labelCount : labelCounts) {
            if (labelCount > 0) {
                double proportion = labelCount / data.numInstances();
                entropy -= (proportion) * log2(proportion);
            }
        }
        return entropy;
    }


    private static double log2(double num) {
        return (num == 0) ? 0 : Math.log(num) / Math.log(2);
    }


    private Instances[] splitData(Instances data, Attribute att) {

        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }

        for (int i = 0; i < data.numInstances(); i++) {
            splitData[(int) data.instance(i).value(att)].add(data.instance(i));
        }

        for (Instances splitData1 : splitData) {
            splitData1.compactify();
        }
        return splitData;
    }
}

