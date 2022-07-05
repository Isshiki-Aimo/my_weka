package weka.classifiers.Aimo;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.util.Comparator;

public class KNN_Aimo extends Classifier {
    //KNN分类器K的个数
    private int k = 10;

    //类别数量
    private int num_classes;

    //属性的数量
    private int num_attributes;

    //样本的数量
    private int num_instaces;

    //类别的标签
    private int class_index;

    //记录每个属性的最小值
    double[] max;
    //记录每个属性的最大值;
    double[] min;

    //存储训练集的数据，用于当测试样本进入分类器计算距离
    private ArrayList<ArrayList<Double>> datam = new ArrayList<>();

    private Instances Set_Instances;        // 实例集合
    private ReplaceMissingValues m_MissingFilter; // 数据预处理需要的过滤器


    public void buildClassifier(Instances data) throws Exception {
        // 备份数据（防止更改原训练数据）
        Set_Instances = new Instances(data);

        // 先做数据预处理（填充缺失值）
        m_MissingFilter = new ReplaceMissingValues();
        m_MissingFilter.setInputFormat(Set_Instances);
        Set_Instances = Filter.useFilter(Set_Instances, m_MissingFilter);
        // 删除没有监督信息的实例
        Set_Instances.deleteWithMissingClass();

        num_classes = Set_Instances.numClasses();
        num_attributes = Set_Instances.numAttributes();
        num_instaces = Set_Instances.numInstances();
        class_index = Set_Instances.classIndex();

        //对训练数据进行存储,按样本进行存储
        for (int i = 0; i < num_instaces; i++) {
            ArrayList<Double> data_single = new ArrayList<>();
            for (int j = 0; j < num_attributes; j++) {
                data_single.add(Set_Instances.instance(i).value(j));
            }
            datam.add(data_single);
        }
        //按属性进行存储
        ArrayList<List<Double>> att = new ArrayList<>();
        for (int i = 0; i < num_attributes; i++) {
                ArrayList<Double> att_single = new ArrayList<>();
                for (int j = 0; j < num_instaces; j++) {
                    att_single.add(Set_Instances.instance(j).value(i));
                }
                att.add(att_single);
        }

        //距离使用欧式距离，对数据进行标准化
        max = new double[num_attributes];
        min = new double[num_attributes];

        for (int i = 0; i < num_attributes; i++) {
            if (i != class_index) {
                att.get(i).sort(Comparator.naturalOrder());//升序排列
                min[i] = att.get(i).get(0);
                att.get(i).sort(Comparator.reverseOrder());//降序排列
                max[i] = att.get(i).get(0);
            }
        }

        //对数据进行处理
        for (int i = 0; i < num_instaces; i++) {
            for (int j = 0; j < num_attributes; j++) {
                if (j != class_index) {
                    double emlent = (datam.get(i).get(j) - min[j]) / (max[j] - min[j]);
                    datam.get(i).set(j, emlent);
                }
            }
        }


    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        // 备份数据
        Instance transformedInstance = new Instance(instance);

        // 先做预处理
        m_MissingFilter.input(transformedInstance);
        m_MissingFilter.batchFinished();
        transformedInstance = m_MissingFilter.output();

        double[] prob = new double[num_classes];//计算该样本属于某个类别的概率
        ArrayList<Double> distance = new ArrayList<>();
        for (int i = 0; i < num_instaces; i++) {
            double dis = 0;
            for (int j = 0; j < num_attributes; j++) {
                if (j != class_index) {
                    double emlent = ((double) transformedInstance.value(j) - min[j]) / (max[j] - min[j]);
                    dis += Math.pow(emlent - datam.get(i).get(j), 2);
                }
            }
            distance.add(Math.sqrt(dis));
        }
        ArrayList<Double> clonedistance = (ArrayList<Double>) distance.clone();
        distance.sort(Comparator.naturalOrder());

        double[] vote = new double[k];
        for (int i = 0; i < k; i++) {
            double a = distance.get(i);
            vote[i] = datam.get(clonedistance.indexOf(a)).get(class_index);
            prob[(int) vote[i]] += 1;
        }

        Utils.normalize(prob);//归一化
        return prob;
    }

    public static void main(String[] argv) {
        try {
            System.out.println(Evaluation.evaluateModel(new KNN_Aimo(), argv));
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println(e.getMessage());
        }
    }
}
