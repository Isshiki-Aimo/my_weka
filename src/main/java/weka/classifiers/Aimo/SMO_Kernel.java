package weka.classifiers.Aimo;


import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.util.Random;

public class SMO_Kernel extends Classifier {
    /**
     * 惩罚因子
     */
    private double C = 1.0;

    /**
     * 松弛变量
     */
    private double tolerance = 0.001;

    /**
     * 终止条件的差值
     */
    private double eps = 0.000000000001;

    /**
     * 训练集的已知类别号，即公式中的y[]
     */
    private double[] y = null;

    /**
     * 训练集的特征向量点,即公式中的x[] <br/>
     * 一行表示一个特征向量，所有行组成训练集的所有特征向量
     */
    private Matrix x = null;

    /**
     * 误差缓存
     */
    private double[] errorCache = null;

    /**
     * 拉格朗日乘数
     */
    private double[] alpha = null;

    /**
     * threshold，阀值
     */
    private double b = 0.0;

    /**
     * rbf kernel for exp(-gamma*|u-v|^2), 默认为0.1，也可设为1/num
     */
    private double gamma = 0.08;

    /**
     * 对points点积的缓存
     */
    private Matrix dotDache = null;

    private double[][] kernel = null;

    /**
     * 所有向量的数目
     */
    private int N;

    private Random random = null;


    private Instances Set_Instances;        // 实例集合
    private ReplaceMissingValues m_MissingFilter; // 数据预处理需要的过滤器

    private int Num_Attributes;              // 属性个数
    private int Num_Instances;              // 实例个数
    private int Num_Classes;                // 类别个数

    private double[] ws;


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

        gamma = 1.0 / Num_Classes;
        x = new Matrix(Num_Instances, Num_Attributes);
        y = new double[Num_Instances];
        N = Num_Instances;

        alpha = new double[N];
        errorCache = new double[N];
        kernel = new double[N][N];
        random = new Random(777);

        for (int i = 0; i < Num_Instances; i++) {
            if (Set_Instances.instance(i).classValue() == 0) {
                y[i] = -1.0;
            } else {
                y[i] = 1.0;
            }
            for (int j = 0; j < Num_Attributes; j++) {
                x.set(i, j, Set_Instances.instance(i).value(j));
            }
        }
        dotDache = x.times(x.transpose());
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
//                kernel[i][j] = Gaussian(dotDache, i, j);
                kernel[i][j] = Polynomial(dotDache, i, j);
//                kernel[i][j] = Linear(dotDache, i, j);
            }
        }
        train();
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
// 备份数据
        Instance transformedInstance = new Instance(instance);

        // 先做预处理
        m_MissingFilter.input(transformedInstance);
        m_MissingFilter.batchFinished();
        transformedInstance = m_MissingFilter.output();
        double[] result;
        double[] test = new double[Num_Attributes];

        for (int i = 0; i < Num_Attributes; i++) {
            test[i] = transformedInstance.value(i);
        }
        result = classify(test, ws);
        return result;
    }


    /**
     * 内循环，选择最大步长的两个点进行优化
     */
    private boolean takeStep(int i1, int i2) {
        if (i1 == i2) {
            return false;
        }

        double alpha1 = alpha[i1];
        double alpha2 = alpha[i2];
        double y1 = y[i1];
        double y2 = y[i2];
        double E1;
        double E2;
        double s = y1 * y2;
        double a1, a2; //新的a
        double L, H;

        if (0 < alpha1 && alpha1 < C) {
            E1 = errorCache[i1];
        } else {
            E1 = calcError(i1);
        }

        if (0 < alpha2 && alpha2 < C) {
            E2 = errorCache[i2];
        } else {
            E2 = calcError(i2);
        }

        if (y1 != y2) {
            L = Math.max(0, alpha2 - alpha1);
            H = Math.min(C, C + alpha2 - alpha1);
        } else {
            L = Math.max(0, alpha1 + alpha2 - C);
            H = Math.min(C, alpha1 + alpha2);
        }
        if (L >= H) {
            return false;
        }

        double k11 = kernel[i1][i1];
        double k12 = kernel[i1][i2];
        double k22 = kernel[i2][i2];

        double eta = 2 * k12 - k11 - k22;
        //根据不同情况计算出a2
        if (eta < 0) {
            //计算非约束条件下的最大值
            a2 = alpha2 - y2 * (E1 - E2) / eta;

            //判断约束的条件
            if (a2 < L) {
                a2 = L;
            } else if (a2 > H) {
                a2 = H;
            }
        } else {
            double C1 = eta / 2;
            double C2 = y2 * (E1 - E2) - eta * alpha2;

            //Lobj和Hobj可以根据自己的爱好选择不同的函数
            double Lobj = C1 * L * L + C2 * L;
            double Hobj = C1 * H * H + C2 * H;

            if (Lobj > Hobj + eps) {
                a2 = L;
            } else if (Lobj < Hobj - eps) {
                a2 = H;
            } else {
                a2 = alpha2;
            }
        }

        if (Math.abs(a2 - alpha2) < eps * (a2 + alpha2 + eps)) {
            return false;
        }

        //通过a2来更新a1
        a1 = alpha1 + s * (alpha2 - a2);

        if (a1 < 0) {
            a2 += s * a1;
            a1 = 0;
        } else if (a1 > C) {
            a2 += s * (a1 - C);
            a1 = C;
        }

        //update threshold b;
        double b1 = b - E1 - y1 * (a1 - alpha1) * k11 - y2 * (a2 - alpha2) * k12;
        double b2 = b - E2 - y1 * (a1 - alpha1) * k12 - y2 * (a2 - alpha2) * k22;

        double bNew;
        if (0 < a1 && a1 < C) {
            bNew = b1;
        } else if (0 < a2 && a2 < C) {
            bNew = b2;
        } else {
            bNew = (b1 + b2) / 2;
        }
        this.b = bNew;


        updateErrorCache(i1);
        updateErrorCache(i2);

        //store a1, a2 in alpha array
        alpha[i1] = a1;
        alpha[i2] = a2;

        return true;
    }


    /**
     * 外循环，检查最好的样本，并进行takeStep计算
     */
    private boolean examineExample(int i1) {
        double y1 = y[i1];
        double alpha1 = alpha[i1];
        double E1;

        if (0 < alpha1 && alpha1 < C) {
            E1 = errorCache[i1];
        } else {
            E1 = calcError(i1);
        }

        double r1 = y1 * E1;
        if ((r1 < -tolerance && alpha1 < C) || (r1 > tolerance && alpha1 > 0)) {

            //选择 E1 - E2 差最大的两点
            int i2 = this.selectMaxJ(E1);
            if (i2 >= 0) {
                if (takeStep(i1, i2)) {
                    return true;
                }
            }

            //先选择 0 < alpha < C的点
            int k0 = randomSelect(i1);
            for (int k = k0; k < N + k0; k++) {
                i2 = k % N;
                if (0 < alpha[i2] && alpha[i2] < C) {
                    if (takeStep(i1, i2)) {
                        return true;
                    }
                }
            }

            //如果不符合，再遍历全部点
            k0 = randomSelect(i1);
            for (int k = k0; k < N + k0; k++) {
                i2 = k % N;
                if (takeStep(i1, i2)) {
                    return true;
                }
            }

        }

        return false;
    }

    public boolean train() {
        int maxIter = 5000;
        int iterCount = 0;
        int numChanged = 0;
        boolean examineAll = true;

        //当迭代次数大于maxIter或者 所有样本中没有alpha对改变时，跳出循环
        while ((iterCount < maxIter) && (numChanged > 0 || examineAll)) {
            numChanged = 0;

            if (examineAll) {
                //循环检查所有样本
                for (int i = 0; i < N; i++) {
                    if (examineExample(i)) {
                        numChanged++;
                    }
                }
            } else {
                //只检查非边界样本
                for (int i = 0; i < N; i++) {
                    if (alpha[i] != 0 && alpha[i] != C) {
                        if (examineExample(i)) {
                            numChanged++;
                        }
                    }
                }
            }

            iterCount++;
            if (examineAll) {
                examineAll = false;
            } else if (numChanged == 0) {
                examineAll = true;
            }
        }
        ws = calcWc();
        return true;
    }


    /**
     * 计算误差公式： error = ∑a[i]*y[i]*k(x,x[i]) - y[i]
     */
    private double calcError(int k) {
        double fxk = 0.0;
        for (int i = 0; i < N; i++) {
            fxk += alpha[i] * y[i] * kernel[i][k];
        }
        fxk += b;
        return fxk - y[k];
    }

    /**
     * 更新误差，重新计算给定点的误差，并保存到errorCache中
     */
    private void updateErrorCache(int k) {
        double error = calcError(k);
        this.errorCache[k] = error;
    }

    /**
     * 找到|E1 - E2|差最大的点的下标
     */
    private int selectMaxJ(double E1) {
        int i2 = -1;
        double tmax = 0.0;
        for (int k = 0; k < N; k++) {
            if (0 < alpha[k] && alpha[k] < C) {
                double E2 = errorCache[k];
                double tmp = Math.abs(E2 - E1);
                if (tmp > tmax) {
                    tmax = tmp;
                    i2 = k;
                }
            }
        }

        return i2;
    }

    /**
     * 随机选择i2，但要求i1 != i2
     */
    private int randomSelect(int i1) {
        int i2;
        do {
            i2 = random.nextInt(N);
        } while (i1 == i2);
        return i2;
    }

    private double[] calcWc() {
        double[] Wc = new double[Num_Attributes];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < Num_Attributes; j++) {
                Wc[j] += alpha[i] * y[i] * x.get(i, j);
            }
        }
        return Wc;
    }

    private double[] classify(double[] X, double[] ws) {
        double prob = 0.0;
        double[] result = new double[2];

        for (int i = 0; i < X.length; i++) {
            prob += ws[i] * X[i];
        }
        prob += b;

        if (prob > 0) {
            result[0] = 0.0;
            result[1] = 1.0;
        } else {
            result[0] = 1.0;
            result[1] = 0.0;
        }
        return result;
    }


    private double Gaussian(Matrix X, int i, int j) {
        return Math.exp(-gamma * (X.get(i, i) + X.get(j, j) - 2 * X.get(i, j)));
    }

    private double Polynomial(Matrix X, int i, int j) {
        return Math.pow(gamma * (X.get(i, i) * X.get(j, j)), 3);
    }

    private double Linear(Matrix X, int i, int j) {
        return X.get(i, i) * X.get(j, j);
    }

}
