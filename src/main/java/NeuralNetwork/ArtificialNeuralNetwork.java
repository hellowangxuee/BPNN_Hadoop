package NeuralNetwork;

import HDFS_IO.ReadNWrite;
import Jampack.*;

import java.io.IOException;
import java.io.Serializable;
import java.nio.DoubleBuffer;
import java.util.Stack;
import java.util.Vector;
import FileIO.*;
import com.sun.org.apache.regexp.internal.RE;
import com.sun.org.apache.xerces.internal.util.URI;
import org.apache.hadoop.metrics.spi.NullContext;
import org.apache.hadoop.yarn.util.StringHelper;
import org.apache.hadoop.yarn.util.SystemClock;

/**
 * Created by Jackie on 16/3/3.
 */
public class ArtificialNeuralNetwork implements Serializable {
    private int LayerNum = 2;
    private NeuronLayer[] ANN = null;
    private Vector TempReult = null;

    public ArtificialNeuralNetwork(NeuronLayer[] NN_arr) {
        this.LayerNum = NN_arr.length;
        this.TempReult = new Vector();
        this.ANN = new NeuronLayer[this.LayerNum];

        for (int i = 0; i < LayerNum; i++) {
            this.ANN[i] = new NeuronLayer(NN_arr[i]);
        }
    }

    public ArtificialNeuralNetwork(int InputNum, int LayerNum, int[] NumEachLayer, int[] IndexEachLayer) {
        this.LayerNum = LayerNum;
        this.ANN = new NeuronLayer[LayerNum];
        this.TempReult = new Vector();

        this.ANN[0] = new NeuronLayer(InputNum, NumEachLayer[0], IndexEachLayer[0]);

        for (int i = 1; i < LayerNum; i++) {
            this.ANN[i] = new NeuronLayer(NumEachLayer[i - 1], NumEachLayer[i], IndexEachLayer[i]);
        }
    }

    public ArtificialNeuralNetwork(String FilePath) throws IOException {
        Vector NNfile = ReadNWrite.hdfs_Read(FilePath);

        this.LayerNum = Integer.parseInt((String) NNfile.get(0));
        this.ANN = new NeuronLayer[this.LayerNum];
        this.TempReult = new Vector();

        int InputNum = Integer.parseInt((String) NNfile.get(1));
        String[] NumEachLayer = (((String) NNfile.get(2)).split("\t"));
        String[] IndexEachLayer = (((String) NNfile.get(3)).split("\t"));

        int i = 4;
        for (int j = 0; j < this.LayerNum; j++) {
            String[][] W_martirx;
            if (j == 0) {
                W_martirx = new String[Integer.parseInt(NumEachLayer[j])][InputNum];
            } else {
                W_martirx = new String[Integer.parseInt(NumEachLayer[j])][Integer.parseInt(NumEachLayer[j - 1])];
            }
            String[] B_martrix = new String[Integer.parseInt(NumEachLayer[j])];
            for (int p = 0; i < NNfile.size(); i++) {
                String[] TextLine = ((String) NNfile.get(i)).split("\t");
                if (TextLine[0].equals("*")) {
                    for (int t = 1; t < TextLine.length; t++) {
                        B_martrix[t - 1] = TextLine[t];
                    }
                } else if (TextLine[0].equals("%")) {
                    i++;
                    break;
                } else {
                    for (int t = 0; t < TextLine.length; t++) {
                        W_martirx[p][t] = TextLine[t];
                    }
                    p += 1;
                }

            }
            this.ANN[j] = new NeuronLayer(W_martirx, B_martrix, IndexEachLayer[j]);
        }
    }

    public double[][] getForwardResult(double[][] InputVec) {
        TempReult.add(InputVec);
        double[][] Result = ANN[0].generateOutput(InputVec);
        TempReult.add(Result);

        for (int i = 1; i < LayerNum; i++) {
            Result = ANN[i].generateOutput(Result);
            TempReult.add(Result);
        }
        return Result;
    }

    public double[][] getForwardResult_singleOutput(String Input) {
        String[] inputArr = Input.split("\t");
        double Tag = Double.parseDouble(inputArr[inputArr.length - 1]);
        double[][] InputVec = new double[inputArr.length - 1][1];
        for (int i = 0; i < inputArr.length - 1; i++) {
            InputVec[i][0] = Double.parseDouble(inputArr[i]);
        }
        TempReult.add(InputVec);
        double[][] Result = ANN[0].generateOutput(InputVec);
        TempReult.add(Result);

        for (int i = 1; i < LayerNum; i++) {
            Result = ANN[i].generateOutput(Result);
            TempReult.add(Result);
        }
        return Result;
    }

    public NeuronLayer[] getBackwardChange(double[][] ErrVec, double LearnRate) {
        try {
            Zmat F;
            Zmat s;
            NeuronLayer[] WeightChangeArr = new NeuronLayer[this.LayerNum];

            F = getMatF(TempReult, LayerNum, ANN[LayerNum - 1].getNeuronNum(), ANN[LayerNum - 1].getTF_index());
            s = Times.o(new Z(-2, 0), Times.o(F, new Zmat(ErrVec)));

            double[][] thisLayerInput = new double[1][ANN[LayerNum - 1].getInputNum()];
            for (int j = 0; j < ANN[LayerNum - 1].getInputNum(); j++) {
                thisLayerInput[0][j] = ((double[][]) (TempReult.get(LayerNum - 1)))[j][0];
            }

            Zmat WeightChange = Times.o(new Z(-LearnRate, 0), Times.o(s, new Zmat(thisLayerInput)));
            Zmat BiasChange = Times.o(new Z(-LearnRate, 0), s);
            NeuronLayer OutputLayerChange = new NeuronLayer(WeightChange, BiasChange, 1);
            WeightChangeArr[LayerNum - 1] = OutputLayerChange;

            for (int i = LayerNum - 2; 0 <= i; i--) {
                F = getMatF(TempReult, i + 1, ANN[i].getNeuronNum(), ANN[i].getTF_index());
                s = Times.o(F, Times.o(transpose.o(new Zmat(ANN[i + 1].getWeightMat())), s));

                thisLayerInput = new double[1][ANN[i].getInputNum()];
                for (int j = 0; j < ANN[i].getInputNum(); j++) {
                    thisLayerInput[0][j] = ((double[][]) (TempReult.get(i)))[j][0];
                }

                WeightChange = Times.o(new Z(-LearnRate, 0), Times.o(s, new Zmat(thisLayerInput)));
                BiasChange = Times.o(new Z(-LearnRate, 0), s);
                NeuronLayer LayerChange = new NeuronLayer(WeightChange, BiasChange, 1);
                WeightChangeArr[i] = LayerChange;
            }
            this.TempReult.clear();
            return WeightChangeArr;
        } catch (Exception e) {
            System.out.println(e.toString());
            return null;
        }

    }

    public NeuronLayer[] getErrorGradient(double[][] ErrVec) {
        try {
            Zmat F;
            Zmat s;
            NeuronLayer[] WeightGradientArr = new NeuronLayer[this.LayerNum];

            F = getMatF(TempReult, LayerNum, ANN[LayerNum - 1].getNeuronNum(), ANN[LayerNum - 1].getTF_index());
            s = Times.o(new Z(-2, 0), Times.o(F, new Zmat(ErrVec)));

            double[][] thisLayerInput = new double[1][ANN[LayerNum - 1].getInputNum()];
            for (int j = 0; j < ANN[LayerNum - 1].getInputNum(); j++) {
                thisLayerInput[0][j] = ((double[][]) (TempReult.get(LayerNum - 1)))[j][0];
            }
            Zmat WeightGradient = Times.o(s, new Zmat(thisLayerInput));
            Zmat BiasGradient = s;
            NeuronLayer OutputLayerGradient = new NeuronLayer(WeightGradient, BiasGradient, 1);
            WeightGradientArr[LayerNum - 1] = OutputLayerGradient;

            for (int i = LayerNum - 2; 0 <= i; i--) {
                F = getMatF(TempReult, i + 1, ANN[i].getNeuronNum(), ANN[i].getTF_index());
                s = Times.o(F, Times.o(transpose.o(new Zmat(ANN[i + 1].getWeightMat())), s));

                thisLayerInput = new double[1][ANN[i].getInputNum()];
                for (int j = 0; j < ANN[i].getInputNum(); j++) {
                    thisLayerInput[0][j] = ((double[][]) (TempReult.get(i)))[j][0];
                }

                WeightGradient = Times.o(s, new Zmat(thisLayerInput));
                BiasGradient = s;
                NeuronLayer LayerGradient = new NeuronLayer(WeightGradient, BiasGradient, 1);
                WeightGradientArr[i] = LayerGradient;
            }
            this.TempReult.clear();
            return WeightGradientArr;
        } catch (Exception e) {
            System.out.println(e.toString());
            return null;
        }
    }

    public Zmat getJacobianMatrix() {
        try {
            Zmat JacobianMatrix;
            Zmat F;
            Zmat MarquardtSensitivity;

            F = getMatF(TempReult, LayerNum, ANN[LayerNum - 1].getNeuronNum(), ANN[LayerNum - 1].getTF_index());
            MarquardtSensitivity = Times.o(new Z(-1, 0), F);

            double[][] thisLayerInput = new double[1][ANN[LayerNum - 1].getInputNum()];
            for (int j = 0; j < ANN[LayerNum - 1].getInputNum(); j++) {
                thisLayerInput[0][j] = ((double[][]) (TempReult.get(LayerNum - 1)))[j][0];
            }

            Zmat WeightJacobian = Times.o(MarquardtSensitivity, new Zmat(thisLayerInput));
            Zmat BiasJacobian = MarquardtSensitivity;
            JacobianMatrix=Merge.o12(WeightJacobian,BiasJacobian);

            for (int i = LayerNum - 2; 0 <= i; i--) {
                F = getMatF(TempReult, i + 1, ANN[i].getNeuronNum(), ANN[i].getTF_index());
                MarquardtSensitivity = Times.o(F, Times.o(transpose.o(new Zmat(ANN[i + 1].getWeightMat())), MarquardtSensitivity));

                thisLayerInput = new double[1][ANN[i].getInputNum()];
                for (int j = 0; j < ANN[i].getInputNum(); j++) {
                    thisLayerInput[0][j] = ((double[][]) (TempReult.get(i)))[j][0];
                }

                WeightJacobian = Times.o(MarquardtSensitivity, new Zmat(thisLayerInput));
                BiasJacobian = MarquardtSensitivity;
                JacobianMatrix = Merge.o13(transpose.o(WeightJacobian), transpose.o(BiasJacobian), JacobianMatrix);
            }
            this.TempReult.clear();
            return JacobianMatrix;
        } catch (Exception e) {
            System.out.println(e.toString());
            return null;
        }
    }

    public boolean updateWeightNetwork(NeuronLayer[] ChangeAmount) {
        try {
            for (int i = 0; i < ANN.length; i++) {
                ANN[i].updateWeightnBias(ChangeAmount[i].getWeightMat(), ChangeAmount[i].getBiasVec());
            }
            return true;
        } catch (Exception e) {
            System.out.println(e.toString());
            return false;
        }
    }

    public boolean updateWeightNetwork(ArtificialNeuralNetwork Another) {
        try {
            for (int i = 0; i < ANN.length; i++) {
                ANN[i].updateWeightnBias(Another.getANN()[i].getWeightMat(), Another.getANN()[i].getBiasVec());
            }
            return true;
        } catch (Exception e) {
            System.out.println(e.toString());
            return false;
        }
    }

    public void updateCertainWeight(int LayNum, int NeuronNum, int WeightNum, double value) {
        ANN[LayNum].updateCertainWeight(NeuronNum, WeightNum, value);
    }

    public void updateCertainBias(int LayNum, int NeuronNum, double value) {
        ANN[LayNum].updateCertainBias(NeuronNum, value);
    }

    public void setCertainWeight(int LayNum, int NeuronNum, int WeightNum, double value) {
        ANN[LayNum].setCertainWeight(NeuronNum, WeightNum, value);
    }

    public void setCertainBias(int LayNum, int NeuronNum, double value) {
        ANN[LayNum].setCertainBias(NeuronNum, value);
    }

    public void clearNetwork() {
        for (int i = 0; i < ANN.length; i++) {
            ANN[i].clearLayer();
        }
    }

    public void averageNetwork(int Q) {
        try {
            for (int i = 0; i < ANN.length; i++) {
                ANN[i].averageLayer(Q);
            }
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    public void multiplyuNetwork(double alpha) {
        for (int i = 0; i < this.LayerNum; i++) {
            for (int j = 0; j < this.ANN[i].getNeuronNum(); j++) {
                this.ANN[i].multiplyCertainNeuron(j, alpha);
            }
            this.ANN[i].multiplyBias(alpha);
        }
    }

    public Zmat getMatF(Vector Re, int IndexNeeded, int OutputNum, int TF_index) {
        Zmat F = new Zmat(OutputNum, OutputNum);

        if (TF_index == 1) {
            double[][] ResultVec = (double[][]) Re.get(IndexNeeded);
            for (int i = 0; i < OutputNum; i++) {
                for (int j = 0; j < OutputNum; j++) {
                    if (i != j) {
                        F.put(i + 1, j + 1, 0);
                    } else {
                        double DeriveOfInput = (1 - ResultVec[i][0]) * (ResultVec[i][0]);
                        F.put(i + 1, j + 1, DeriveOfInput);
                    }

                }
            }
        } else if (TF_index == 3) {
            for (int i = 0; i < OutputNum; i++) {
                for (int j = 0; j < OutputNum; j++) {
                    if (i != j) {
                        F.put(i + 1, j + 1, 0);
                    } else {
                        F.put(i + 1, j + 1, 1);
                    }

                }
            }
        } else if (TF_index == 4) {
            double[][] ResultVec = (double[][]) Re.get(IndexNeeded);
            for (int i = 0; i < OutputNum; i++) {
                for (int j = 0; j < OutputNum; j++) {
                    if (i != j) {
                        F.put(i + 1, j + 1, 0);
                    } else {
                        double DeriveOfInput = 1 / (1 / (Math.exp(ResultVec[i][0]) - 1) + 1);
                        F.put(i + 1, j + 1, DeriveOfInput);
                    }
                }
            }
        }

        return F;
    }

    public NeuronLayer[] getANN() {
        return this.ANN;
    }

    public int getLayerNum() {
        return this.LayerNum;
    }

    public int getOutputNum(){
        return this.ANN[this.LayerNum-1].getNeuronNum();
    }

    public int getInputNum(){
        return this.ANN[0].getInputNum();
    }

    public String[] saveANN() {
        int totalLineNum = 4;
        for (int i = 0; i < this.LayerNum; i++) {
            totalLineNum += this.ANN[i].getNeuronNum() + 2;
        }
        String[] ANN_content_arr = new String[totalLineNum];
        for (int i = 0; i < ANN_content_arr.length; i++) {
            ANN_content_arr[i] = "";
        }

        ANN_content_arr[0] = String.valueOf(this.LayerNum);
        ANN_content_arr[1] = String.valueOf(this.ANN[0].getInputNum());
        for (int i = 0; i < this.LayerNum; i++) {
            ANN_content_arr[2] += String.valueOf(this.ANN[i].getNeuronNum()) + "\t";
            ANN_content_arr[3] += String.valueOf(this.ANN[i].getTF_index()) + "\t";
        }
        int currentLineNum = 4;
        for (int i = 0; i < this.LayerNum; i++) {
            for (int j = 0; j < this.ANN[i].getNeuronNum(); j++) {
                for (int k = 0; k < this.ANN[i].getInputNum(); k++) {
                    ANN_content_arr[currentLineNum] += String.valueOf(this.ANN[i].getCertainWeight(j, k)) + "\t";
                }
                currentLineNum++;
            }
            ANN_content_arr[currentLineNum] += "*\t";
            for (int j = 0; j < this.ANN[i].getNeuronNum(); j++) {
                ANN_content_arr[currentLineNum] += String.valueOf(this.ANN[i].getCertainBias(j)) + "\t";
            }
            currentLineNum++;
            ANN_content_arr[currentLineNum] = "%";
            currentLineNum++;
        }
        return ANN_content_arr;
    }

    public static NeuronLayer[] getFR_CGD(NeuronLayer[] ThisIteGradientArr, NeuronLayer[] LastIteGradientArr, NeuronLayer[] LastIteCGDirection) {
        try {
            NeuronLayer[] CG_Direction = new NeuronLayer[ThisIteGradientArr.length];
            for (int i = 0; i < ThisIteGradientArr.length; i++) {
                for (int j = 0; j < ThisIteGradientArr[i].getNeuronNum(); j++) {
                    double tmp1 = 0.0;
                    double tmp2 = 0.0;
                    for (int k = 0; k < ThisIteGradientArr[i].getInputNum(); k++) {
                        tmp1 += (ThisIteGradientArr[i].getCertainWeight(j, k) * ThisIteGradientArr[i].getCertainWeight(j, k));
                        tmp2 += (LastIteGradientArr[i].getCertainWeight(j, k) * LastIteGradientArr[i].getCertainWeight(j, k));
                    }
                    LastIteCGDirection[i].multiplyCertainNeuron(j, (tmp1 / tmp2));
                }
                double sum1 = 0;
                double sum2 = 0;
                for (int k = 0; k < ThisIteGradientArr[i].getNeuronNum(); k++) {
                    sum1 += (ThisIteGradientArr[i].getCertainBias(k) * ThisIteGradientArr[i].getCertainBias(k));
                    sum2 += (LastIteGradientArr[i].getCertainBias(k) * LastIteGradientArr[i].getCertainBias(k));
                }
                LastIteCGDirection[i].multiplyBias(sum1 / sum2);

                CG_Direction[i] = NeuronLayer.getSubtractionBetweenTwo(LastIteCGDirection[i], ThisIteGradientArr[i]);
            }
            return CG_Direction;
        } catch (Exception e) {
            System.out.println(e.toString());
            return null;
        }
    }

    public static NeuronLayer[] getPR_CGD(NeuronLayer[] ThisIteGradientArr, NeuronLayer[] LastIteGradientArr, NeuronLayer[] LastIteCGDirection) {
        try {
            NeuronLayer[] CG_Direction = new NeuronLayer[ThisIteGradientArr.length];
            for (int i = 0; i < ThisIteGradientArr.length; i++) {
                for (int j = 0; j < ThisIteGradientArr[i].getNeuronNum(); j++) {
                    double tmp1 = 0.0;
                    double tmp2 = 0.0;
                    for (int k = 0; k < ThisIteGradientArr[i].getInputNum(); k++) {
                        tmp1 += ((ThisIteGradientArr[i].getCertainWeight(j, k) - LastIteGradientArr[i].getCertainWeight(j, k)) * ThisIteGradientArr[i].getCertainWeight(j, k));
                        tmp2 += (LastIteGradientArr[i].getCertainWeight(j, k) * LastIteGradientArr[i].getCertainWeight(j, k));
                    }
                    LastIteCGDirection[i].multiplyCertainNeuron(j, (tmp1 / tmp2));
                }
                double sum1 = 0;
                double sum2 = 0;
                for (int k = 0; k < ThisIteGradientArr[i].getNeuronNum(); k++) {
                    sum1 += ((ThisIteGradientArr[i].getCertainBias(k) - LastIteGradientArr[i].getCertainBias(k)) * ThisIteGradientArr[i].getCertainBias(k));
                    sum2 += (LastIteGradientArr[i].getCertainBias(k) * LastIteGradientArr[i].getCertainBias(k));
                }
                LastIteCGDirection[i].multiplyBias(sum1 / sum2);

                CG_Direction[i] = NeuronLayer.getSubtractionBetweenTwo(LastIteCGDirection[i], ThisIteGradientArr[i]);
            }
            return CG_Direction;
        } catch (Exception e) {
            System.out.println(e.toString());
            return null;
        }
    }

    public static ArtificialNeuralNetwork addTwoANN(NeuronLayer[] ANN1, NeuronLayer[] ANN2) {
        NeuronLayer[] NewNeuronLayerArr = new NeuronLayer[ANN1.length];

        for (int i = 0; i < NewNeuronLayerArr.length; i++) {
            NewNeuronLayerArr[i] = NeuronLayer.getAdditionBetweenTwo(ANN1[i], ANN2[i]);
        }

        ArtificialNeuralNetwork ResultANN = new ArtificialNeuralNetwork(NewNeuronLayerArr);
        return ResultANN;
    }

    public static NeuronLayer[] multiplyNeuronLayers(NeuronLayer[] NL, double alpha) {
        NeuronLayer[] MNL = new NeuronLayer[NL.length];

        for (int i = 0; i < NL.length; i++) {
            MNL[i] = NeuronLayer.getMultiplyBasedOne(NL[i], alpha);
        }

        return MNL;
    }

    public static double getNeuronLayerArrLength_norm2(NeuronLayer[] ANN){
        double Length=0;
        for(int i=0;i<ANN.length;i++){
            for(int j=0;j<ANN[i].getNeuronNum();j++){
                for(int k=0;k<ANN[i].getInputNum();k++){
                    Length+= ANN[i].getCertainWeight(j,k) *ANN[i].getCertainWeight(j,k);
                }
                Length+=ANN[i].getCertainBias(j) * ANN[i].getCertainBias(j);
            }
        }
        return Math.sqrt(Length);
    }
}
