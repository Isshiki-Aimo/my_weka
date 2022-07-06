package weka.classifiers.Aimo;

import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.core.*;
import weka.core.Capabilities.Capability;


public class AODE_Aimo
        extends Classifier
        implements OptionHandler, WeightedInstancesHandler, UpdateableClassifier {


    private double[][][] m_CondiCounts;

    private double[] m_ClassCounts;

    private double[][] m_SumForCounts;

    private int m_NumClasses;


    private int m_NumAttributes;


    private int m_NumInstances;


    private int m_ClassIndex;

    private Instances m_Instances;

    private int m_TotalAttValues;


    private int[] m_StartAttIndex;


    private int[] m_NumAttValues;


    private double[] m_Frequencies;


    private double m_SumInstances;


    private int m_Limit = 1;

    private boolean m_MEstimates = false;

    private int m_Weight = 1;

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);

        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        result.setMinimumNumberInstances(0);

        return result;
    }

    public void buildClassifier(Instances instances) throws Exception {

        // 判断数据集是否可用
        getCapabilities().testWithFail(instances);

        // 删除有缺失值的数据集
        m_Instances = new Instances(instances);
        m_Instances.deleteWithMissingClass();


        m_SumInstances = 0;
        m_ClassIndex = instances.classIndex();
        m_NumInstances = m_Instances.numInstances();
        m_NumAttributes = m_Instances.numAttributes();
        m_NumClasses = m_Instances.numClasses();

        m_StartAttIndex = new int[m_NumAttributes];
        m_NumAttValues = new int[m_NumAttributes];

        m_TotalAttValues = 0;
        for (int i = 0; i < m_NumAttributes; i++) {
            if (i != m_ClassIndex) {
                m_StartAttIndex[i] = m_TotalAttValues;
                m_NumAttValues[i] = m_Instances.attribute(i).numValues();
                m_TotalAttValues += m_NumAttValues[i] + 1;
            } else {
                m_NumAttValues[i] = m_NumClasses;
            }
        }

        m_CondiCounts = new double[m_NumClasses][m_TotalAttValues][m_TotalAttValues];
        m_ClassCounts = new double[m_NumClasses];
        m_SumForCounts = new double[m_NumClasses][m_NumAttributes];
        m_Frequencies = new double[m_TotalAttValues];

        for (int k = 0; k < m_NumInstances; k++) {
            addToCounts((Instance) m_Instances.instance(k));
        }

        m_Instances = new Instances(m_Instances, 0);
    }


    public void updateClassifier(Instance instance) {
        this.addToCounts(instance);
    }

    private void addToCounts(Instance instance) {

        double[] countsPointer;

        if (instance.classIsMissing())
            return;

        int classVal = (int) instance.classValue();
        double weight = instance.weight();

        m_ClassCounts[classVal] += weight;
        m_SumInstances += weight;

        int[] attIndex = new int[m_NumAttributes];
        for (int i = 0; i < m_NumAttributes; i++) {
            if (i == m_ClassIndex)
                attIndex[i] = -1;
            else {
                if (instance.isMissing(i))
                    attIndex[i] = m_StartAttIndex[i] + m_NumAttValues[i];
                else
                    attIndex[i] = m_StartAttIndex[i] + (int) instance.value(i);
            }
        }

        for (int Att1 = 0; Att1 < m_NumAttributes; Att1++) {
            if (attIndex[Att1] == -1)
                continue;

            m_Frequencies[attIndex[Att1]] += weight;

            if (!instance.isMissing(Att1))
                m_SumForCounts[classVal][Att1] += weight;

            countsPointer = m_CondiCounts[classVal][attIndex[Att1]];

            for (int Att2 = 0; Att2 < m_NumAttributes; Att2++) {
                if (attIndex[Att2] != -1) {
                    countsPointer[attIndex[Att2]] += weight;
                }
            }
        }
    }


    public double[] distributionForInstance(Instance instance) throws Exception {

        double[] probs = new double[m_NumClasses];

        int pIndex, parentCount;

        double[][] countsForClass;
        double[] countsForClassParent;

        int[] attIndex = new int[m_NumAttributes];
        for (int att = 0; att < m_NumAttributes; att++) {
            if (instance.isMissing(att) || att == m_ClassIndex)
                attIndex[att] = -1;
            else
                attIndex[att] = m_StartAttIndex[att] + (int) instance.value(att);
        }

        for (int classVal = 0; classVal < m_NumClasses; classVal++) {

            probs[classVal] = 0;
            double spodeP;
            parentCount = 0;
            countsForClass = m_CondiCounts[classVal];
            for (int parent = 0; parent < m_NumAttributes; parent++) {
                if (attIndex[parent] == -1)
                    continue;
                pIndex = attIndex[parent];
                if (m_Frequencies[pIndex] < m_Limit)
                    continue;
                countsForClassParent = countsForClass[pIndex];
                attIndex[parent] = -1;
                parentCount++;
                double classparentfreq = countsForClassParent[pIndex];
                double missing4ParentAtt =
                        m_Frequencies[m_StartAttIndex[parent] + m_NumAttValues[parent]];
                if (!m_MEstimates) {
                    spodeP = (classparentfreq + 1.0)
                            / ((m_SumInstances - missing4ParentAtt) + m_NumClasses
                            * m_NumAttValues[parent]);
                } else {
                    spodeP = (classparentfreq + ((double) m_Weight
                            / (double) (m_NumClasses * m_NumAttValues[parent])))
                            / ((m_SumInstances - missing4ParentAtt) + m_Weight);
                }
                for (int att = 0; att < m_NumAttributes; att++) {
                    if (attIndex[att] == -1)
                        continue;

                    double missingForParentandChildAtt =
                            countsForClassParent[m_StartAttIndex[att] + m_NumAttValues[att]];

                    if (!m_MEstimates) {
                        spodeP *= (countsForClassParent[attIndex[att]] + 1.0)
                                / ((classparentfreq - missingForParentandChildAtt)
                                + m_NumAttValues[att]);
                    } else {
                        spodeP *= (countsForClassParent[attIndex[att]]
                                + ((double) m_Weight / (double) m_NumAttValues[att]))
                                / ((classparentfreq - missingForParentandChildAtt)
                                + m_Weight);
                    }
                }
                probs[classVal] += spodeP;
                attIndex[parent] = pIndex;
            }
            if (parentCount < 1) {
                probs[classVal] = NBconditionalProb(instance, classVal);

            } else {
                probs[classVal] /= parentCount;
            }
        }
        Utils.normalize(probs);
        return probs;
    }


    public double NBconditionalProb(Instance instance, int classVal) {

        double prob;
        double[][] pointer;

        if (!m_MEstimates) {
            prob = (m_ClassCounts[classVal] + 1.0) / (m_SumInstances + m_NumClasses);
        } else {
            prob = (m_ClassCounts[classVal]
                    + ((double) m_Weight / (double) m_NumClasses))
                    / (m_SumInstances + m_Weight);
        }
        pointer = m_CondiCounts[classVal];

        for (int att = 0; att < m_NumAttributes; att++) {
            if (att == m_ClassIndex || instance.isMissing(att))
                continue;

            int aIndex = m_StartAttIndex[att] + (int) instance.value(att);

            if (!m_MEstimates) {
                prob *= (pointer[aIndex][aIndex] + 1.0)
                        / (m_SumForCounts[classVal][att] + m_NumAttValues[att]);
            } else {
                prob *= (pointer[aIndex][aIndex]
                        + ((double) m_Weight / (double) m_NumAttValues[att]))
                        / (m_SumForCounts[classVal][att] + m_Weight);
            }
        }
        return prob;
    }

}
