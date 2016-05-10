package TEP_Classification;

import FileIO.FileReadNWrite;
import Jampack.*;
import NeuralNetwork.ArtificialNeuralNetwork;
import org.apache.hadoop.yarn.webapp.hamlet.Hamlet;

import java.io.IOException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Vector;

import static FileIO.FileReadNWrite.readTxtFile;

/**
 * Created by mlx on 5/10/16.
 */
public class TEP_LMBP {
    public static void main(String[] args) throws JampackException, IOException {
        Vector<Double[]> TrainData = readTxtFile("/home/mlx/Documents/DataSet/TEP_TrainData");
        Vector<Double[]> TestData = readTxtFile("/home/mlx/Documents/DataSet/TEP_TestData");

        ArtificialNeuralNetwork FinalANN = null;
        int InputNum = 41;
        int LayerNum = 3;
        int[] NumEachLayer = {7, 10, 1};
        int[] IndexEachLayer = {1, 4, 4};
        double MSE_upbound = 0.01;

        String[] RunningMessage = new String[10000];
        int ParaNum = 0;
        FinalANN = new ArtificialNeuralNetwork(InputNum, LayerNum, NumEachLayer, IndexEachLayer);
        for (int i = 0; i < FinalANN.getLayerNum(); i++) {
            ParaNum += FinalANN.getANN()[i].getNeuronNum() * (FinalANN.getANN()[i].getInputNum() + 1);
        }
        double[][] ErrVec = new double[NumEachLayer[NumEachLayer.length - 1]][1];
        double[][] ForwardResult;
        double[][] ErrArr = new double[TrainData.size()][1];

        Zmat TotalJacobianMatrix = null;

        System.out.println(String.valueOf(ParaNum));
        System.out.println("TryTime" + "\t" + "Ite" + "\t" + "MSE\tmiu");

        double ThisErrorSum = 0.0;
        double LastErrorSum = 0.0;
        double miu = 0.1;
        ArtificialNeuralNetwork UpdatesANN = null;
        try {
            Zmat LastJacobMatrix = null;
            Zmat[] JacobMatrixSeries = new Zmat[TrainData.size()];
            for (int Ite = 0; Ite < 2000; Ite++) {
                double SquareErr = 0.0;
                TotalJacobianMatrix = null;
                for (int i = 0; i < TrainData.size(); i++) {
                    Double[] temp = (TrainData.get(i));
                    double[][] InputVec = new double[InputNum][1];
                    for (int k = 0; k < temp.length - 1; k++) {
                        InputVec[k][0] = temp[k];
                    }
                    double Tag = temp[temp.length - 1];

                    ForwardResult = FinalANN.getForwardResult(InputVec);
                    ErrVec[0][0] = Tag - ForwardResult[0][0];

                    SquareErr += Math.pow(ErrVec[0][0], 2);
                    ErrArr[i][0] = ErrVec[0][0];
                    JacobMatrixSeries[i] = FinalANN.getJacobianMatrix();
//                    if (i == 0) {
//                        TotalJacobianMatrix = FinalANN.getJacobianMatrix();
//                    } else {
//                        try {
//                            TotalJacobianMatrix = Merge.o21(TotalJacobianMatrix, FinalANN.getJacobianMatrix());
//                        } catch (Exception e) {
//                            System.out.println(e.toString());
//                            continue;
//                        }
//                    }
                }
                ThisErrorSum = SquareErr;

                if (ThisErrorSum / TrainData.size() < MSE_upbound) {
                    break;
                } else if (Ite > 0) {
                    if (ThisErrorSum < LastErrorSum) {
                        miu /= 10;
                        LastErrorSum = ThisErrorSum;
                        TotalJacobianMatrix = new Zmat(TrainData.size(), ParaNum);
                        for (int k = 0; k < TrainData.size(); k++) {
                            for (int p = 0; p < ParaNum; p++) {
                                TotalJacobianMatrix.put0(k, p, new Z(JacobMatrixSeries[k].get0(0, p)));
                            }
                        }
                        LastJacobMatrix = TotalJacobianMatrix;
                    } else {
                        miu *= 10;
                        FinalANN.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(UpdatesANN.getANN(), -1.0));
                        TotalJacobianMatrix = LastJacobMatrix;
                    }
                } else {
                    LastErrorSum = ThisErrorSum;
                    TotalJacobianMatrix = new Zmat(TrainData.size(), ParaNum);
                    for (int k = 0; k < TrainData.size(); k++) {
                        for (int p = 0; p < ParaNum; p++) {
                            TotalJacobianMatrix.put0(k, p, new Z(JacobMatrixSeries[k].get0(0, p)));
                        }
                    }
                    LastJacobMatrix = TotalJacobianMatrix;
                }
                Zmat H = Times.o(transpose.o(TotalJacobianMatrix), TotalJacobianMatrix);
//                    Eig EigOfH = new Eig(H);
//                    double minEigValue = EigOfH.D.get0(H.nr - 1).re;
//                    if (minEigValue < 0 && miu + minEigValue < 0) {
//                        miu += (-minEigValue);
//                    }
                Zmat G = Plus.o(H, Times.o(new Z(miu, 0), Eye.o(H.nr)));
                Zmat NetworkUpdates = Times.o(new Z(-1, 0), Times.o(Times.o(Inv.o(G), transpose.o(TotalJacobianMatrix)), new Zmat(ErrArr)));
                UpdatesANN = new ArtificialNeuralNetwork(FinalANN.getANN());
                int LayerIndex = 0;
                int NeuronIndex = 0;
                int InputIndex = 0;
                for (int index = 0; index < NetworkUpdates.nr; index++) {
                    int LayerParaSum = 0;
                    for (int i = 0; i < UpdatesANN.getLayerNum(); i++) {
                        LayerParaSum += UpdatesANN.getANN()[i].getNeuronNum() * (UpdatesANN.getANN()[i].getInputNum() + 1);
                        if (index / LayerParaSum < 1) {
                            LayerIndex = i;
                            break;
                        }
                    }
                    int NeuronOffset = index - LayerParaSum + UpdatesANN.getANN()[LayerIndex].getNeuronNum() * (UpdatesANN.getANN()[LayerIndex].getInputNum() + 1);
                    if (NeuronOffset >= (UpdatesANN.getANN()[LayerIndex].getNeuronNum() * UpdatesANN.getANN()[LayerIndex].getInputNum())) {
                        int BiasNum = NeuronOffset - (UpdatesANN.getANN()[LayerIndex].getNeuronNum() * UpdatesANN.getANN()[LayerIndex].getInputNum());
                        UpdatesANN.setCertainBias(LayerIndex, BiasNum, NetworkUpdates.get0(index, 0).re);
                    } else {
                        NeuronIndex = NeuronOffset / UpdatesANN.getANN()[LayerIndex].getInputNum();
                        InputIndex = NeuronOffset % UpdatesANN.getANN()[LayerIndex].getInputNum();
                        UpdatesANN.setCertainWeight(LayerIndex, NeuronIndex, InputIndex, NetworkUpdates.get0(index, 0).re);
                    }
                }
                double LengthOfUpdate = ArtificialNeuralNetwork.getNeuronLayerArrLength_norm2(UpdatesANN.getANN());
                System.out.println(String.valueOf(Ite) + "\t" + String.valueOf(SquareErr / TrainData.size()) + "\t" + miu + "\t" + LengthOfUpdate);
                RunningMessage[Ite] = String.valueOf(Ite) + "\t" + String.valueOf(SquareErr / TrainData.size()) + "\t" + String.valueOf(LengthOfUpdate);
                FinalANN.updateWeightNetwork(UpdatesANN);
            }
        } catch (Exception e) {
            System.out.println(e.toString());
        }

        FileReadNWrite.LocalWriteFile("/home/mlx/Documents/ProcessResult/TEP_LMBP_IteMsg", RunningMessage);

        String[] TestResult = new String[TestData.size()];
        double[][] TestForwardResult;
        double TestErrSum = 0.0;
        for (int i = 0; i < TestData.size(); i++) {
            Double[] temp = (TestData.get(i));
            double[][] InputVec = new double[InputNum][1];
            for (int k = 0; k < temp.length - 1; k++) {
                InputVec[k][0] = temp[k];
            }
            double Tag = temp[temp.length - 1];

            TestForwardResult = FinalANN.getForwardResult(InputVec);
            TestResult[i] = String.valueOf(TestForwardResult[0][0]);
            TestErrSum += (Tag - TestForwardResult[0][0]) * (Tag - TestForwardResult[0][0]);
        }
        TestErrSum /= TestData.size();
        System.out.println("\n" + String.valueOf(TestErrSum));

        String[] SaveANN = FinalANN.saveANN();
        FileReadNWrite.LocalWriteFile("/home/mlx/Documents/ResultANN/TEP_LMBP_FinalANN", SaveANN);
        FileReadNWrite.LocalWriteFile("/home/mlx/Documents/ProcessResult/TEP_LMBP_PreRe", TestResult);
    }

    public static int getRandomNum(int m,int n) {
        return (m + (int) (Math.random() * (n - m + 1)));
    }
}