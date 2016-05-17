package StandAloneNN;

import FileIO.FileReadNWrite;
import Jampack.*;
import NeuralNetwork.ArtificialNeuralNetwork;

import java.io.IOException;
import java.util.HashSet;
import java.util.Vector;

import static FileIO.FileReadNWrite.readTxtFile;

/**
 * Created by mlx on 5/7/16.
 */
public class SA_LMBP {
    public static void main(String[] args) throws JampackException, IOException {
        Vector<Double[]> TrainData = readTxtFile("/home/mlx/Documents/DataSet/StandAloneFuncTrainData_WithNoise");
        Vector<Double[]> TestData = readTxtFile("/home/mlx/Documents/DataSet/StandAloneFuncTestData");

        ArtificialNeuralNetwork FinalANN = null;
        int InputNum = 1;
        int LayerNum = 3;
        int[] NumEachLayer = {5, 7, 1};
        int[] IndexEachLayer = {1, 1, 3};
        double MSE_upbound=0.001;

        boolean IfFindSolution = false;
        for (int TryTime=0;!IfFindSolution;TryTime++) {
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
            double miu = 0.001;
            ArtificialNeuralNetwork UpdatesANN = null;
            try {
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
                        if (i == 0) {
                            TotalJacobianMatrix = FinalANN.getJacobianMatrix();
                        } else {
                            try {
                                TotalJacobianMatrix = Merge.o21(TotalJacobianMatrix, FinalANN.getJacobianMatrix());
                            } catch (Exception e) {
                                System.out.println(e.toString());
                                continue;
                            }
                        }
                    }
                    ThisErrorSum = SquareErr;

                    System.out.println(String.valueOf(TryTime) + "\t" + String.valueOf(Ite) + "\t" + String.valueOf(SquareErr / TrainData.size())+"\t"+miu);
                    if (ThisErrorSum / TrainData.size() < MSE_upbound) {
                        IfFindSolution = true;
                        break;
                    } else if (Ite > 0) {
                        if (ThisErrorSum < LastErrorSum) {
                            miu /= 10;
                            LastErrorSum = ThisErrorSum;
                        } else {
                            miu *= 10;
                            FinalANN.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(UpdatesANN.getANN(), -1.0));
                        }
                    } else {
                        LastErrorSum = ThisErrorSum;
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
                    FinalANN.updateWeightNetwork(UpdatesANN);

                }
            } catch (Exception e) {
                System.out.println(e.toString());
            }
        }

        double[][] ForwardResult;
        double TestErrSum=0.0;
        for (int i = 0; i < TestData.size(); i++) {
            Double[] temp = (TestData.get(i));
            double[][] InputVec = new double[InputNum][1];
            for (int k = 0; k < temp.length - 1; k++) {
                InputVec[k][0] = temp[k];
            }
            double Tag = temp[temp.length - 1];

            ForwardResult = FinalANN.getForwardResult(InputVec);
            System.out.println(ForwardResult[0][0]);
            TestErrSum+=(Tag-ForwardResult[0][0])*(Tag-ForwardResult[0][0]);
        }
        TestErrSum/=TestData.size();
        System.out.println("\n"+TestErrSum);
        String[] SaveANN = FinalANN.saveANN();
        FileReadNWrite.LocalWriteFile("/home/mlx/Documents/ResultANN/LMBP_FinalANN", SaveANN);
    }
}
