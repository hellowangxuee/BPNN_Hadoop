package NeuralNetwork;

import java.nio.DoubleBuffer;

import Jampack.JampackException;
import Jampack.Zmat;
import Jampack.mult;
/**
 * Created by Jackie on 16/3/3.
 */
public class SingleNeuron {
    private int InputNum = 0;
    private double[][] WeightVector = null;
    private double Bias = 0.0;
    private int TF_index = 1;

    public SingleNeuron(int InputNum, int TF_index) {
        this.InputNum = InputNum;
        this.WeightVector = new double[1][InputNum];

        for (int i = 0; i < InputNum; i++) {
            this.WeightVector[0][i] = Math.random() * 0.1 + 0.01;
        }
        this.Bias = Math.random() * 0.1 + 0.01;
        this.TF_index = TF_index;
    }

    public double generateOutput(double[][] InputVec) throws JampackException {
        try {
            double result = 0.0;
            Zmat matInput = new Zmat(InputVec);
            Zmat matWeight = new Zmat(this.WeightVector);

            result = mult.o(matWeight, matInput).re[0][0] + Bias;

            if (TF_index == 1) {
                result = TransFunc.Log_Sigmoid(result);
            } else if (TF_index == 2) {
                result = TransFunc.Symmetrical_Hard_Limit(result);
            } else if (TF_index == 3) {
                result = TransFunc.Pure_Linear(result);
            } else {
                result = TransFunc.ReLu(result);
            }
            return result;
        } catch (Exception e) {
            System.out.println(e.toString());
            return 0.0;
        }
    }

    public void updateWeightnBias(double[][] NewWeight, double NewBias) {
        for (int i = 0; i < WeightVector[0].length; i++) {
            WeightVector[0][i] += NewWeight[0][i];
        }
        Bias += NewBias;
    }

    public int getInputNum() {
        return this.InputNum;
    }

    public int getTF_index() {
        return this.TF_index;
    }

    public double getBias() {
        return this.Bias;
    }

    public double getCertainWeight(int j) {
        return this.WeightVector[0][j];
    }

    public double[][] WeightVector() {
        return this.WeightVector;
    }

}
