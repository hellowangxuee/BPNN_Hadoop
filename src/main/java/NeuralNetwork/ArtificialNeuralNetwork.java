package NeuralNetwork;

import HDFS_IO.ReadNWrite;
import Jampack.*;

import java.io.IOException;
import java.io.Serializable;
import java.nio.DoubleBuffer;
import java.util.Vector;
import FileIO.*;
import org.apache.hadoop.yarn.util.StringHelper;

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
                String[] TextLine = ((String)NNfile.get(i)).split("\t");
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

    public double[][] getForwardResult_singleOutput(String Input){
        String[] inputArr=Input.split("\t");
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
                thisLayerInput[0][j] = ((double[][]) (TempReult.get(LayerNum - 1)))[j][0];//   TempReult[LayerNum - 1].get(j);
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
}
