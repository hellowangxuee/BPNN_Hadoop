package StandAloneNN;

import Jampack.*;
import NeuralNetwork.ArtificialNeuralNetwork;
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec;

import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Vector;

import static FileIO.FileReadNWrite.readTxtFile;

/**
 * Created by mlx on 4/20/16.
 */
public class BayesianRegularization_LMBP {
    public static void main(String[] args) throws JampackException, IOException {
        Vector<Double[]> TrainData = readTxtFile("/home/mlx/Documents/DataSet/TrainDataSet3");
        Vector<Double[]> TestData = readTxtFile("/home/mlx/Documents/DataSet/TestDataSet3");

        ArtificialNeuralNetwork FinalANN=null ;
        boolean IfFindSolution=false;
        while (!IfFindSolution) {
            double BETA = 1;
            double ALPHA = 0.01;
            double GAMMA = 0.0;
            int ParaNum = 0;

            int InputNum = 1;
            int LayerNum = 3;
            int[] NumEachLayer = {10, 10, 1};
            int[] IndexEachLayer = {1, 1, 3};

            FinalANN = new ArtificialNeuralNetwork(InputNum, LayerNum, NumEachLayer, IndexEachLayer);

            for (int i = 0; i < FinalANN.getLayerNum(); i++) {
                ParaNum += FinalANN.getANN()[i].getNeuronNum() * (FinalANN.getANN()[i].getInputNum() + 1);
            }


            double[][] ErrVec = new double[NumEachLayer[NumEachLayer.length - 1]][1];
            double[][] ForwardResult;

            double[][] ErrArr = new double[TrainData.size()][1];
            Zmat TotalJacobianMatrix = null;
            GAMMA = ParaNum;

            System.out.println(String.valueOf(ParaNum));
            System.out.println("Ite" + "\t" + "ALPHA" + "\t" + "BETA" + "\t" + "ALPHA / BETA" + "\t" + "GAMMA" + "\t" + "SquareErr/TrainData.size()");

            HashSet h = new HashSet();
            try {
                for (int Ite = 0; Ite < 1000; Ite++) {
                    double SquareErr = 0.0;
                    double WeightSquareSum = FinalANN.getWeightSquareSum();

                    int RandomNum = getRandomNum(0, TrainData.size() - 1);
                    h.add(RandomNum);
                    for (int i = 0; i < TrainData.size(); i++) {
                        while (!h.contains(RandomNum)) {
                            RandomNum = getRandomNum(0, TrainData.size() - 1);
                            h.add(RandomNum);
                        }
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
                    h.clear();

                    ALPHA = GAMMA / (2 * WeightSquareSum);
                    BETA = (TrainData.size() - GAMMA) / (2 * SquareErr);
                    Zmat H = Times.o(transpose.o(TotalJacobianMatrix), TotalJacobianMatrix);
                    Zmat G = Plus.o(Times.o(new Z(2 * BETA, 0), H), Times.o(new Z(2 * ALPHA, 0), Eye.o(H.nr)));
                    GAMMA = ParaNum - 2 * ALPHA / Trace.o((G)).re;

                    double miu = 0.01;
                    Eig EigOfH = new Eig(H);
                    double minEigValue = 2 * BETA * EigOfH.D.get0(H.nr - 1).re + 2 * ALPHA;
                    if (minEigValue < 0 && miu + minEigValue < 0) {
                        miu += (-minEigValue);
                    }
                    G = Plus.o(G, Times.o(new Z(miu, 0), Eye.o(H.nr)));

                    Zmat NetworkUpdates = Times.o(new Z(-1, 0), Times.o(Times.o(Inv.o(G), transpose.o(TotalJacobianMatrix)), new Zmat(ErrArr)));
                    ArtificialNeuralNetwork UpdatesANN = new ArtificialNeuralNetwork(FinalANN.getANN());
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
                    FinalANN.updateWeightNetwork(UpdatesANN.getANN());
                    System.out.println(String.valueOf(Ite) + "\t" + String.valueOf(ALPHA) + "\t" + String.valueOf(BETA) + "\t" + String.valueOf(ALPHA / BETA) + "\t" + String.valueOf(GAMMA) + "\t" + String.valueOf(SquareErr / TrainData.size()));
                    if (SquareErr / TrainData.size() < 0.005) {
                        IfFindSolution = true;
                        break;
                    }
                }
            }
            catch (Exception e){

            }
        }

        int InputNum=1;
        double[][] ForwardResult;

        for (int i = 0; i < TestData.size(); i++) {
            Double[] temp = (TestData.get(i));
            double[][] InputVec = new double[InputNum][1];
            for (int k = 0; k < temp.length - 1; k++) {
                InputVec[k][0] = temp[k];
            }
            double Tag = temp[temp.length - 1];

            ForwardResult = FinalANN.getForwardResult(InputVec);
            System.out.println(ForwardResult[0][0]);
        }
    }

    public static int getRandomNum(int m,int n) {
        return (m + (int) (Math.random() * (n - m + 1)));
    }
}
