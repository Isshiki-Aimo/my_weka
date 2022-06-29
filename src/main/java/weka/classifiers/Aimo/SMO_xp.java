package weka.classifiers.Aimo;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;

import java.util.Random;

public class SMO_xp extends Classifier {
    //类别数目
    private int num_class;
    //属性个数
    private int num_attribute;
    //样本个数
    private int num_instances;
    //类别标签
    private int class_index;
    //定义软间隔，允许一些样本出错
    private double C=0.6;
    //容错率，用来调节软间隔
    private double toler=0.001;
    //存在每个样本点的alpha值
    private Matrix alphas;
    //阈值b
    private double b;
    //存放每个样本点的误差
    private Matrix error;
    //存放真实标签
    private Matrix target;
    //存放样本数据
    private Matrix input;
    //设置迭代次数
    private int max_iter=60;
    //计算W
    private Matrix W;


    public final Random random=new Random();

    public void buildClassifier(Instances data) throws Exception{
        num_class=data.numClasses();
        num_instances=data.numInstances();
        num_attribute=data.numAttributes();
        class_index=data.classIndex();
        alphas=new Matrix(num_instances,1);
        error=new Matrix(num_instances,2);//第一列进行判断，该误差是否有效，第二列存放有效值
        target=new Matrix(num_instances,1);
        input=new Matrix(num_instances,num_attribute-1);//有一种属性作为类别
        W=new Matrix(num_attribute-1,1);
        for (int i=0;i<num_instances;i++)
        {
            double randomValue=random.nextDouble();
            alphas.set(i,0,randomValue);
            boolean flag=false;
            for (int j=0;j<num_attribute;j++)
            {
                if (j!=class_index)
                {
                    if (flag==false)
                    {
                        input.set(i,j,data.instance(i).value(j));
                    }
                    if (flag==true)
                    {
                        input.set(i,j-1,data.instance(i).value(j));
                    }
                }
                if (j==class_index)
                {
                    flag=true;
                    if(data.instance(i).value(j)==1)
                    {
                        target.set(i,0,data.instance(i).value(j));
                    }
                    if(data.instance(i).value(j)==0)
                    {
                        target.set(i,0,-1);
                    }

                }
            }
        }

        for (int i=0;i<max_iter;i++)
        {
            for (int j=0;j<num_instances;j++)
            {
                update(j);
            }
        }
        calculate_W();
    }

    public int selectrand(int i)
    {
        //随机生成一个下标，作为alpha2的索引
        int j=0;
        j=random.nextInt(num_instances);
        while (j==i)
        {
            j=random.nextInt(num_instances);
        }
        return j;
    }

    public double cutAlpha(double aj,double up,double low)
    {
        //经分析，alphai是有范围的，0<=aj<=C,所以对获取的aj进行截取
        if (aj>up)
        {
            aj=up;
        }
        if (aj<low)
        {
            aj=low;
        }
        return aj;
    }

    public double calculateerror(int k)
    {
        //计算第k个样本的误差值
        double ek=0.0;
        Matrix aiyi=alphas.arrayTimes(target);//逐个元素相乘，（n_instance,1）
        double fx=0.0;
        for(int i=0;i<num_instances;i++)
        {
            double a=input.getMatrix(k,k,0,num_attribute-2).times(input.getMatrix(i,i,0,num_attribute-2).transpose()).get(0,0);//(1,1)矩阵，然后取出来
            fx+=aiyi.get(i,0)*a;
        }
        fx+=b;
        ek=fx-target.get(k,0);
        return ek;
    }

    public int select_maxj(int i)
    {
        //获取具有最大差值|ei-ej|的j
        //后续可以来不全，完整版SMOaj的选取采用最大化步长，简单版就随机选取，先用简单版
        int j=0;
        return j;
    }

    public void update(int i)
    {
        double eai=calculateerror(i);
        int j=selectrand(i);
        double eaj=calculateerror(j);

        double alphai_old=alphas.get(i,0);
        double alphaj_old=alphas.get(j,0);

        double up=0.0;
        double low=0.0;


        //分析约束条件，确定alpha的up和lower，进行截断
        if(target.get(i,0)!=target.get(j,0))
        {
            if (alphaj_old-alphai_old>0)
            {
                low=alphaj_old-alphai_old;
            }
            else {
                low=0.0;
            }
            if (C<C+alphaj_old-alphai_old)
            {
                up=C;
            }
            else {
                up=C+alphaj_old-alphai_old;
            }
        }

        if (target.get(i,0)==target.get(j,0))
        {
            if (alphaj_old+alphai_old-C>0)
            {
                low=alphaj_old+alphai_old-C;
            }
            else {
                low=0.0;
            }
            if (C<alphaj_old+alphai_old)
            {
                up=C;
            }
            else {
                up=alphaj_old+alphai_old;
            }
        }

        //计算eta=K11+K22-2*K12
        double eta=input.getMatrix(i,i,0,num_attribute-2).times(input.getMatrix(i,i,0,num_attribute-2).transpose()).get(0,0)
                +input.getMatrix(j,j,0,num_attribute-2).times(input.getMatrix(j,j,0,num_attribute-2).transpose()).get(0,0)
                -2*input.getMatrix(i,i,0,num_attribute-2).times(input.getMatrix(j,j,0,num_attribute-2).transpose()).get(0,0);

        //更新alpha2
        alphas.set(j,0,alphas.get(j,0)+target.get(j,0)*(eai-eaj)/eta);
        alphas.set(j,0,cutAlpha(alphas.get(j,0),up,low));
        eaj=calculateerror(j);

        //更新alpha1
        alphas.set(i,0,alphas.get(i,0)+target.get(i,0)*target.get(j,0)*(alphaj_old-alphas.get(j,0)));
        eai=calculateerror(i);


        //更新阈值b
        double b1=-eai-target.get(i,0)*input.getMatrix(j,j,0,num_attribute-2).times(input.getMatrix(i,i,0,num_attribute-2).transpose()).get(0,0)
                *(alphas.get(i,0)-alphai_old)-target.get(j,0)*input.getMatrix(j,j,0,num_attribute-2).times(input.getMatrix(i,i,0,num_attribute-2).transpose()).get(0,0)
                *(alphas.get(j,0)-alphaj_old)+b;
        double b2=-eaj-target.get(i,0)*input.getMatrix(i,i,0,num_attribute-2).times(input.getMatrix(j,j,0,num_attribute-2).transpose()).get(0,0)
                *(alphas.get(i,0)-alphai_old)-target.get(j,0)*input.getMatrix(j,j,0,num_attribute-2).times(input.getMatrix(j,j,0,num_attribute-2).transpose()).get(0,0)
                *(alphas.get(j,0)-alphaj_old)+b;
        b=(b1+b2)/2.0;
    }

    public void calculate_W()
    {
        for(int j=0;j<num_attribute-1;j++)
        {
            for (int i=0;i<num_instances;i++)
            {
                W.set(j,0,W.get(j,0)+alphas.get(i,0)*target.get(i,0)*input.get(i,j));
            }
        }
    }



    public double[] distributionForInstance(Instance instance) throws Exception{
        double []prob=new double[num_class];
        boolean flag=false;
        double fx=0.0;
        for (int j=0;j<num_attribute;j++)
        {
            if (j!=class_index)
            {
                if (flag==false)
                {
                    fx+=instance.value(j)*W.get(j,0);
                }
                if (flag==true)
                {
                    fx+=instance.value(j)* W.get(j-1,0);
                }
            }
            if (j==class_index)
            {
                flag=true;
            }
        }
        fx+=b;
        if (fx>=0)
        {
            prob[0]=0;
            prob[1]=1;
        }
        if (fx<=0)
        {
            prob[0]=1;
            prob[1]=0;
        }
        return prob;
    }




}
