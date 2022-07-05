package weka.classifiers.Aimo;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.util.ArrayList;
import java.util.Random;


public class SMO_Aimo extends Classifier {

    class optStruct {
        private Matrix X;
        private Matrix label;
        private double C;
        private double tol;
        private int m;
        private Matrix alpha;
        private Matrix b;
        private Matrix eCache;

        public optStruct(Matrix dataMatIn, Matrix classLabels, double C, double toler) {
            X = dataMatIn;
            label = classLabels;
            this.C = C;
            tol = toler;
            m = X.getRowDimension();
            alpha = new Matrix(m, 1);
            b = new Matrix(1, 1);
            eCache = new Matrix(m, 2);
        }
    }

    Random random = new Random(777777);

    private Instances Set_Instances;        // 实例集合
    private ReplaceMissingValues m_MissingFilter; // 数据预处理需要的过滤器

    private int Num_Attributes;              // 属性个数
    private int Num_Instances;              // 实例个数
    private int Num_Classes;                // 类别个数
    private double gamma = 0.08;
    Matrix test;
    Matrix b;

    Matrix ws;

    @Override
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

        Matrix Matrix_X = new Matrix(Num_Instances, Num_Attributes);
        Matrix Matrix_Y = new Matrix(Num_Instances, 1);


        for (int i = 0; i < Num_Instances; i++) {
            if (Set_Instances.instance(i).classValue() == 0) {
                Matrix_Y.set(i, 0, -1);
            } else {
                Matrix_Y.set(i, 0, 1);
            }
            for (int j = 0; j < Num_Attributes; j++) {
                Matrix_X.set(i, j, Set_Instances.instance(i).value(j));
            }
        }

        optStruct optStruct = new optStruct(Matrix_X, Matrix_Y, 0.7, 0.0001);
        smoP(optStruct, 40);
        b = optStruct.b;

        ws = calcWs(optStruct.alpha, optStruct.X, optStruct.label);

    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
// 备份数据
        Instance transformedInstance = new Instance(instance);

        // 先做预处理
        m_MissingFilter.input(transformedInstance);
        m_MissingFilter.batchFinished();
        transformedInstance = m_MissingFilter.output();
        double[] result;

        test = new Matrix(1, Num_Attributes);
        for (int i = 0; i < Num_Attributes; i++) {
            test.set(0, i, transformedInstance.value(i));
        }
        result = classify(test, ws, b);
        return result;
    }


    //随机选取一个数J,为后面内循环选取α2的值做准备
    private int selectJrand(int i, int m) {
        int j = i;
        while (j == i) {
            j = random.nextInt(m);
        }
        return j;
    }

    //对αj进行截取操作
    private double clipAlpha(double aj, double H, double L) {
        if (aj > H) {
            return H;
        } else return Math.max(aj, L);
    }

    //计算Ei，即误差Ei=f(xi)-yi
    private double calcEk(optStruct oS, int k) {
        double fxk = oS.alpha.arrayTimes(oS.label).transpose()
                .times(oS.X.times(oS.X.getMatrix(k, k, 0, oS.X.getColumnDimension() - 1).transpose()))
                .plus(oS.b).get(0, 0);
        return fxk - oS.label.get(k, 0);
    }

    //内循环的启发式方法，获取最大差值|Ei-Ej|的αj和Ej
    private Matrix selectJ(optStruct oS, int i, double Ei) {
        int maxK = -1;
        double maxDelta = 0;    //保存临时最大差值
        double Ej = 0;
        Matrix Ej_arr = new Matrix(2, 1);
        oS.eCache.set(i, 0, 1);
        oS.eCache.set(i, 1, Ei);

        ArrayList<Integer> validEcacheList = new ArrayList<Integer>();
        for (int k = 0; k < oS.eCache.getRowDimension(); k++) {
            if (oS.eCache.get(k, 0) != 0) {
                validEcacheList.add(k);
            }
        }

        //如果有不为0的Ei，则从中选取最大的Ej
        if (validEcacheList.size() > 1) {
            for (int k = 0; k < validEcacheList.size(); k++) {
                if (validEcacheList.get(k) == i) {
                    continue;
                }
                //开始计算Ek，进行对比获取最大差值
                double Ek = calcEk(oS, k);
                double deltaE = Math.abs(Ei - Ek);
                if (deltaE > maxDelta) {
                    //更新最大差值和最大差值对应的Ej
                    maxDelta = deltaE;
                    maxK = k;
                    Ej = Ek;
                    Ej_arr.set(0, 0, maxK);
                    Ej_arr.set(1, 0, Ej);
                }
            }
        } else//如果没有有效误差缓存，则随机选取一个索引，进行返回
        {
            int j = selectJrand(i, oS.m);
            Ej = calcEk(oS, j);
            Ej_arr.set(0, 0, j);
            Ej_arr.set(1, 0, Ej);
        }
        return Ej_arr;
    }

    // 更新Ek操作
    private void updateEk(optStruct oS, int k) {
        double Ek = calcEk(oS, k);
        oS.eCache.set(k, 0, 1);
        oS.eCache.set(k, 1, Ek);
    }

    //实现内循环函数
    private int inner(int i, optStruct oS) {
        double Ei = calcEk(oS, i);
        double L, H;
        //如果下面违背了KKT条件，进行正常的α、Ek、b的更新
        if ((oS.label.get(i, 0) * Ei < -oS.tol && oS.alpha.get(i, 0) < oS.C)
                || (oS.label.get(i, 0) * Ei > oS.tol && oS.alpha.get(i, 0) > 0)) {
            Matrix Ej_arr = selectJ(oS, i, Ei);
            int j = (int) Ej_arr.get(0, 0);
            double Ej = Ej_arr.get(1, 0);
            //保存更新前的alpha值，用于更新后的对比
            double alphaIold = oS.alpha.get(i, 0);
            double alphaJold = oS.alpha.get(j, 0);

            //计算上下界L和H
            if (oS.label.get(i, 0) != oS.label.get(j, 0)) {
                L = Math.max(0, alphaJold - alphaIold);
                H = Math.min(oS.C, oS.C + alphaJold - alphaIold);
            } else {
                L = Math.max(0, alphaJold + alphaIold - oS.C);
                H = Math.min(oS.C, alphaJold + alphaIold);
            }
            if (L == H) {
                return 0;
            }
            //计算η=k11+k22-2k12
            double eta = oS.X.getMatrix(i, i, 0, oS.X.getColumnDimension() - 1).
                    times(oS.X.getMatrix(i, i, 0, oS.X.getColumnDimension() - 1).transpose()).
                    plus((oS.X.getMatrix(j, j, 0, oS.X.getColumnDimension() - 1).
                            times(oS.X.getMatrix(j, j, 0, oS.X.getColumnDimension() - 1).transpose())
                    )).minus(oS.X.getMatrix(i, i, 0, oS.X.getColumnDimension() - 1).
                            times(oS.X.getMatrix(j, j, 0, oS.X.getColumnDimension() - 1).transpose())
                            .times(2.0)).get(0, 0);
            if (eta <= 0) {
                return 0;
            }
            //当满足上述条件时，更新α2值，并更新对应的Ek值
            oS.alpha.set(j, 0, oS.alpha.get(j, 0) + oS.label.get(j, 0) * (Ei - Ej) / eta);
            oS.alpha.set(j, 0, clipAlpha(oS.alpha.get(j, 0), H, L));
            updateEk(oS, j);

//查看α_2是否有足够的变化量，如果没有足够变化量，我们直接返回，不进行下面更新α_1,注意：因为α_2变化量较小，所以我们没有必要非得把值变回原来的旧值
            if (Math.abs(oS.alpha.get(j, 0) - alphaJold) < 0.00001) {
                return 0;
            }
            //开始更新α_1值,和Ek值
            oS.alpha.set(i, 0, oS.alpha.get(i, 0) +
                    oS.label.get(i, 0) * oS.label.get(j, 0)
                            * (alphaJold - oS.alpha.get(j, 0)));
            updateEk(oS, i);
//开始更新阈值b,正好使用到了上面更新的Ek值
            double b1 = oS.b.get(0, 0) - Ei - oS.label.get(i, 0) *
                    (oS.alpha.get(i, 0) - alphaIold) *
                    (oS.X.getMatrix(i, i, 0, oS.X.getColumnDimension() - 1).
                            times(oS.X.getMatrix(i, i, 0, oS.X.getColumnDimension() - 1).transpose()).get(0, 0))
                    - oS.label.get(j, 0) * (oS.alpha.get(j, 0) - alphaJold) *
                    (oS.X.getMatrix(i, i, 0, oS.X.getColumnDimension() - 1).
                            times(oS.X.getMatrix(j, j, 0, oS.X.getColumnDimension() - 1).transpose()).get(0, 0));


            double b2 = oS.b.get(0, 0) - Ej - oS.label.get(i, 0) *
                    (oS.alpha.get(i, 0) - alphaIold) *
                    (oS.X.getMatrix(i, i, 0, oS.X.getColumnDimension() - 1).
                            times(oS.X.getMatrix(j, j, 0, oS.X.getColumnDimension() - 1).transpose()).get(0, 0))
                    - oS.label.get(j, 0) * (oS.alpha.get(j, 0) - alphaJold) *
                    (oS.X.getMatrix(j, j, 0, oS.X.getColumnDimension() - 1).
                            times(oS.X.getMatrix(j, j, 0, oS.X.getColumnDimension() - 1).transpose()).get(0, 0));

//根据统计学习方法中阈值b在每一步中都会进行更新，
//        1.当新值alpha_1不在界上时(0<alpha_1<C)，b_new的计算规则为：b_new=b1
//        2.当新值alpha_2不在界上时(0 < alpha_2 < C)，b_new的计算规则为：b_new = b2
//        3.否则当alpha_1和alpha_2都不在界上时，b_new = 1/2(b1+b2)
            if (oS.alpha.get(i, 0) > 0 && oS.alpha.get(i, 0) < oS.C) {
                oS.b.set(0, 0, b1);
            } else if (oS.alpha.get(j, 0) > 0 && oS.alpha.get(j, 0) < oS.C) {
                oS.b.set(0, 0, b2);
            } else {
                oS.b.set(0, 0, (b1 + b2) / 2.0);
            }
            return 1;
        }
        return 0;
    }

    private void smoP(optStruct oS, int maxIter) {
        int i = 0;
        int alphaPairsChanged = 0;
        int iter = 0;
        boolean entireSet = true;
        //后面的判断条件中，表示上一次循环中，是在整个数据集中遍历，并且没有α值更新过，则退出
        while ((iter < maxIter) && (alphaPairsChanged > 0) || entireSet) {
            alphaPairsChanged = 0;
            if (entireSet) {
                for (i = 0; i < oS.m; i++) {
                    alphaPairsChanged += inner(i, oS);
                }
            } else {
                ArrayList<Integer> nonBounds = new ArrayList<>();
                for (int k = 0; k < oS.label.getRowDimension(); k++) {
                    if (oS.alpha.get(k, 0) > 0 && oS.alpha.get(k, 0) < oS.C) {
                        nonBounds.add(k);
                    }
                }
                for (Integer nonBound : nonBounds) {
                    alphaPairsChanged += inner(nonBound, oS);
                }
            }
            iter++;
            if (entireSet)
                entireSet = false;
            else if (alphaPairsChanged == 0) //如果是在非边界上，并且α更新过。则entireSet还是False,
            // 下一次还是在非边界上进行遍历。可以认为这里是倾向于非边界遍历，因为非边界遍历的样本更符合内循环中的违反KKT条件
            {
                entireSet = true;
            }
        }
    }

    private Matrix calcWs(Matrix alpha, Matrix dataArr, Matrix labelArr) {
        Matrix w = new Matrix(Num_Attributes, 1);
        for (int i = 0; i < Num_Instances; i++) {
            w = w.plus(dataArr.getMatrix(i, i, 0, dataArr.getColumnDimension() - 1).transpose().
                    times(alpha.get(i, 0) * labelArr.get(i, 0)));
        }
        return w;
    }


    private double[] classify(Matrix X, Matrix ws, Matrix b) {
        Matrix prob;
        double[] result = new double[2];
        prob = X.times(ws).plus(b);
        if (prob.get(0, 0) > 0) {
            result[0] = 0.0;
            result[1] = 1.0;
            return result;
        } else {
            result[0] = 1.0;
            result[1] = 0.0;
            return result;
        }
    }

}
