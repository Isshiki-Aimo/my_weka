package weka.classifiers.Aimo;

import weka.core.matrix.Matrix;


public class My_BDG {
    private int Num_Attributes = 2;                // 属性个数
    private int Num_Instances;            // 实例个数

    // x = m * d
    //thetas = d * 1

    private Matrix Compute_Gradient(final Matrix x, final Matrix y, final Matrix theta, int Batch_size) {
        Matrix gradient = new Matrix(Num_Attributes, 1);
        Matrix h_theta;

        h_theta = x.times(theta);

        for (int i = 0; i < Num_Attributes; i++) {
            //求一个属性的梯度
            double sum = 0;
            for (int j = 0; j < Batch_size; j++) {
                double a = ((h_theta.get(j, 0) - y.get(j, 0)) * x.get(j, i));
                sum += a;
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

    public static void main(String[] args) {
        My_BDG bdg = new My_BDG();
        bdg.Num_Attributes = 2;
        bdg.Num_Instances = 3;
        Matrix x = new Matrix(bdg.Num_Instances, bdg.Num_Attributes);
        x.set(0, 0, 1d);
        x.set(1, 0, 2d);
        x.set(2, 0, 3d);
        x.set(0, 1, 1d);
        x.set(1, 1, 1d);
        x.set(2, 1, 1d);
        Matrix y = new Matrix(bdg.Num_Instances, 1);
        y.set(0, 0, 5d);
        y.set(1, 0, 9d);
        y.set(2, 0, 13d);
        Matrix thetas = bdg.Gradient_Descent(0.01, x, y, 0.0000001, 100000);
        System.out.println(thetas);
    }

}