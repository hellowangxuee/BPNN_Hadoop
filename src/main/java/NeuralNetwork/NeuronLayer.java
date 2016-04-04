package NeuralNetwork;

import Jampack.JampackException;
import Jampack.Plus;
import Jampack.Zmat;
import Jampack.mult;
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec;

/**
 * Created by Jackie on 16/3/3.
 */
public class NeuronLayer {
    private int InputNum = 0;
    private double[][] WeightMat = null;
    private double[][] BiasVec = null;
    private int TF_index = 1;

    public NeuronLayer(int InputNum,int OutputNum,int TF_index) {
        this.InputNum = InputNum;
        this.TF_index = TF_index;

        this.WeightMat = new double[OutputNum][InputNum];
        this.BiasVec = new double[OutputNum][1];

        for (int i = 0; i < OutputNum; i++) {
            for (int j = 0; j < InputNum; j++) {
                this.WeightMat[i][j] = Math.random()-0.5;
            }
            this.BiasVec[i][0] = Math.random()-0.5;
        }
    }


    public NeuronLayer(SingleNeuron[] NeuronArr) {
        this.InputNum = NeuronArr[0].getInputNum();
        this.TF_index = NeuronArr[0].getTF_index();

        this.WeightMat = new double[NeuronArr.length][InputNum];
        this.BiasVec = new double[InputNum][1];

        for (int i = 0; i < NeuronArr.length; i++) {
            for (int j = 0; j < NeuronArr[i].getInputNum(); j++) {
                this.WeightMat[i][j] = NeuronArr[i].getCertainWeight(j);
            }
            this.BiasVec[i][0] = NeuronArr[i].getBias();
        }
    }

    public NeuronLayer(NeuronLayer ALayer) {
        this.InputNum = ALayer.getInputNum();
        this.TF_index = ALayer.getTF_index();

        int NeronNum=ALayer.getWeightMat().length;

        this.WeightMat = new double[NeronNum][InputNum];
        this.BiasVec = new double[NeronNum][1];

        for (int i = 0; i < NeronNum; i++) {
            for (int j = 0; j < InputNum; j++) {
                this.WeightMat[i][j] = ALayer.getCertainWeight(i, j);
            }
            this.BiasVec[i][0] = ALayer.getCertainBias(i);
        }
    }

    public NeuronLayer(Zmat W,Zmat B,int index){
        this.InputNum= W.nc;
        this.TF_index=index;

        this.WeightMat=new double[W.nr][W.nc];
        this.BiasVec=new double[B.nr][B.nc];

        for(int i=0;i<W.nr;i++){
            for(int j=0;j<W.nc;j++){
                this.WeightMat[i][j]=W.get(i+1,j+1).re;
            }
            this.BiasVec[i][0]= B.get(i+1,1).re;
        }

    }

    public NeuronLayer(String[][] W,String[] B,String index){
        this.InputNum=W[0].length;
        this.TF_index=Integer.parseInt(index);

        this.WeightMat=new double[W.length][W[0].length];
        this.BiasVec=new double[B.length][1];

        for(int i=0;i<W.length;i++){
            for(int j=0;j<W[0].length;j++){
                this.WeightMat[i][j]=Double.parseDouble(W[i][j]) ;
            }
            this.BiasVec[i][0]= Double.parseDouble(B[i]);
        }
    }



    public double[][] generateOutput(double[][] InputVec) {
        try {
            Zmat matInput = new Zmat(InputVec);
            Zmat matWeight = new Zmat(WeightMat);
            Zmat matBias = new Zmat(BiasVec);

            Zmat Result = Plus.o(mult.o(matWeight, matInput), matBias);

            double[][] Output = new double[Result.nr][1];

            if (TF_index == 1) {
                for (int i = 0; i < Result.nr; i++) {
                    Output[i][0] = TransFunc.Log_Sigmoid(Result.re[i][0]);
                }
            } else if (TF_index == 2) {
                for (int i = 0; i < Result.nr; i++) {
                    Output[i][0] = TransFunc.Symmetrical_Hard_Limit(Result.re[i][0]);
                }
            } else if (TF_index == 3) {
                for (int i = 0; i < Result.nr; i++) {
                    Output[i][0] = TransFunc.Pure_Linear(Result.re[i][0]);
                }
            } else {
                for (int i = 0; i < Result.nr; i++) {
                    Output[i][0] = TransFunc.ReLu(Result.re[i][0]);
                }
            }

            return Output;
        } catch (Exception e) {
            System.out.println(e.toString());
            return null;
        }
    }

    public void updateWeightnBias(double[][] NewWeight, double[][] NewBias) {
        for (int i = 0; i < WeightMat.length; i++) {
            for (int j = 0; j < WeightMat[i].length; j++) {
                WeightMat[i][j] += NewWeight[i][j];
            }
            BiasVec[i][0] += NewBias[i][0];
        }
    }

    public void updateCertainWeight(int i,int j,double value) {
        WeightMat[i][j] += value;
    }

    public void updateCertainBias(int i,double value) {
        BiasVec[i][0] += value;
    }

    public void setCertainWeight(int i,int j,double value) {
        WeightMat[i][j] = value;
    }

    public void setCertainBias(int i,double value) {
        BiasVec[i][0] = value;
    }

    public void clearLayer(){
        for (int i = 0; i < WeightMat.length; i++) {
            for (int j = 0; j < WeightMat[i].length; j++) {
                WeightMat[i][j] =0;
            }
            BiasVec[i][0] =0;
        }
    }

    public void averageLayer(int Q){
        if(Q!=0) {
            for (int i = 0; i < WeightMat.length; i++) {
                for (int j = 0; j < WeightMat[i].length; j++) {
                    WeightMat[i][j] /= Q;
                }
                BiasVec[i][0] /= Q;
            }
        }
    }

    public int getInputNum() {
        return this.InputNum;
    }

    public int getTF_index() {
        return this.TF_index;
    }

    public double[][] getWeightMat() {
        return this.WeightMat;
    }

    public double[][] getBiasVec() {
        return this.BiasVec;
    }

    public double getCertainWeight(int i, int j) {
        return this.WeightMat[i][j];
    }

    public double getCertainBias(int i) {
        return this.BiasVec[i][0];
    }

    public int getNeuronNum(){
        return this.WeightMat.length;
    }

    public double getMaxElement() {
        double maxEle = 0.0;
        for (int i = 0; i < this.WeightMat.length; i++) {
            for (int j = 0; j < this.InputNum; j++) {
                if (Math.abs(this.WeightMat[i][j]) > maxEle) {
                    maxEle = Math.abs(this.WeightMat[i][j]);
                }
            }
            if (Math.abs(this.BiasVec[i][0]) > maxEle) {
                maxEle = Math.abs(this.BiasVec[i][0]);
            }
        }
        return maxEle;
    }

    public void multiplyCertainNeuron(int NeuronNum,double beta) {
        if (beta != 0) {
            for (int j = 0; j < this.InputNum; j++) {
                this.WeightMat[NeuronNum][j] *= beta;
            }
        }
    }

    public void multiplyBias(double beta) {
        if (beta != 0) {
            for (int j = 0; j < this.WeightMat.length; j++) {
                this.BiasVec[j][0] *= beta;
            }
        }
    }

    public void multiplyNeuronLayer(double alpha){
        if(alpha!=0){
            for(int i=0;i<this.WeightMat.length;i++){
                for(int j=0;j<this.InputNum;j++){
                    this.WeightMat[i][j] *= alpha;
                }
                this.BiasVec[i][0] *=alpha;
            }
        }
    }

    public static NeuronLayer getSubtractionBetweenTwo(NeuronLayer N1,NeuronLayer N2) {
        NeuronLayer SubResult = new NeuronLayer(N1.getInputNum(), N1.getNeuronNum(), N1.getTF_index());
        for (int i = 0; i < N1.getNeuronNum(); i++) {
            for (int j = 0; j < N1.getInputNum(); j++) {
                SubResult.setCertainWeight(i, j, N1.getCertainWeight(i, j) - N2.getCertainWeight(i, j));
            }
            SubResult.setCertainBias(i, N1.getCertainBias(i) - N2.getCertainBias(i));
        }
        return SubResult;
    }

    public static NeuronLayer getAdditionBetweenTwo(NeuronLayer N1,NeuronLayer N2) {
        NeuronLayer AddResult = new NeuronLayer(N1.getInputNum(), N1.getNeuronNum(), N1.getTF_index());
        for (int i = 0; i < N1.getNeuronNum(); i++) {
            for (int j = 0; j < N1.getInputNum(); j++) {
                AddResult.setCertainWeight(i, j, N1.getCertainWeight(i, j) + N2.getCertainWeight(i, j));
            }
            AddResult.setCertainBias(i, N1.getCertainBias(i) + N2.getCertainBias(i));
        }
        return AddResult;
    }

    public static NeuronLayer getMultiplyBasedOne(NeuronLayer N1,double alpha) {
        NeuronLayer MulResult = new NeuronLayer(N1.getInputNum(), N1.getNeuronNum(), N1.getTF_index());
        for (int i = 0; i < N1.getNeuronNum(); i++) {
            for (int j = 0; j < N1.getInputNum(); j++) {
                MulResult.setCertainWeight(i, j, N1.getCertainWeight(i, j) * alpha);
            }
            MulResult.setCertainBias(i, N1.getCertainBias(i) * alpha);
        }
        return MulResult;
    }
}
