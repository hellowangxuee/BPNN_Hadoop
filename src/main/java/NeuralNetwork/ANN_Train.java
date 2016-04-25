package NeuralNetwork;

import Jampack.*;
import com.sun.org.apache.xml.internal.security.Init;
import org.apache.commons.math3.stat.descriptive.summary.Sum;

import java.util.Vector;

/**
 * Created by mlx on 4/25/16.
 */
public class ANN_Train {
    public static ArtificialNeuralNetwork BayReg_BPTrain(Vector<Double[]> TrainingData,int InputNum,int LayerNum,int[] NeuNumEachLayer,int[] NeuIndexEachLayer,double MSE_UpBound) {
        boolean IfFindSolution = false;
        int ParaNum = 0;
        for (int i = 0; i < NeuNumEachLayer.length; i++) {
            if (i == 0) {
                ParaNum += NeuNumEachLayer[i] * (InputNum + 1);
            } else {
                ParaNum += NeuNumEachLayer[i] * (NeuNumEachLayer[i - 1] + 1);
            }
        }

        ArtificialNeuralNetwork FinalANN = null;
        for (int TryTime=0; !IfFindSolution && TryTime<5; TryTime++) {
            double BETA = 1;
            double ALPHA = 0.01;
            double GAMMA = ParaNum;

            FinalANN = new ArtificialNeuralNetwork(InputNum, LayerNum, NeuNumEachLayer, NeuIndexEachLayer);

            double[][] ErrVec = new double[1][1];
            double[][] ForwardResult = null;
            double[][] ErrorArr = new double[TrainingData.size()][1];
            Zmat TotalJacobianMatrix = null;
            try {
                for (int Ite = 0; Ite < 1000; Ite++) {
                    double SquareErrSum = 0.0;
                    double WeightSquareSum = FinalANN.getWeightSquareSum();
                    for (int i = 0; i < TrainingData.size(); i++) {
                        Double[] SingleEntry = TrainingData.get(i);
                        double[][] InputVec = new double[InputNum][1];
                        for (int k = 0; k < SingleEntry.length - 1; k++) {
                            InputVec[k][0] = SingleEntry[k];
                        }
                        double EntryTag = SingleEntry[SingleEntry.length - 1];
                        ForwardResult = FinalANN.getForwardResult(InputVec);
                        ErrVec[0][0] = EntryTag - ForwardResult[0][0];
                        SquareErrSum += Math.pow(ErrVec[0][0], 2);
                        ErrorArr[i][0] = ErrVec[0][0];
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

                    ALPHA = GAMMA / (2 * WeightSquareSum);
                    BETA = (TrainingData.size() - GAMMA) / (2 * SquareErrSum);
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

                    Zmat NetworkUpdates = Times.o(new Z(-1, 0), Times.o(Times.o(Inv.o(G), transpose.o(TotalJacobianMatrix)), new Zmat(ErrorArr)));
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
                    if (SquareErrSum / TrainingData.size() < MSE_UpBound) {
                        IfFindSolution = true;
                        break;
                    }
                }
            } catch (Exception e) {

            }

        }
        return FinalANN;
    }

    public static ArtificialNeuralNetwork BayReg_BPTrain(Vector<Double[]> TrainingData,ArtificialNeuralNetwork InitialANN,double MSE_UpBound) {
        int ParaNum = 0;
        for (int i = 0; i < InitialANN.getLayerNum(); i++) {
            ParaNum += InitialANN.getANN()[i].getNeuronNum() * (InitialANN.getANN()[i].getInputNum() + 1);
        }

        double BETA = 1;
        double ALPHA = 0.01;
        double GAMMA = ParaNum;

        ArtificialNeuralNetwork FinalANN = new ArtificialNeuralNetwork(InitialANN.getANN());
        ArtificialNeuralNetwork SumUpdatesANN = new ArtificialNeuralNetwork(InitialANN.getANN());
        SumUpdatesANN.clearNetwork();

        double[][] ErrVec = new double[1][1];
        double[][] ForwardResult = null;
        double[][] ErrorArr = new double[TrainingData.size()][1];
        Zmat TotalJacobianMatrix = null;
        try {
            for (int Ite = 0; Ite < 150; Ite++) {
                double SquareErrSum = 0.0;
                double WeightSquareSum = FinalANN.getWeightSquareSum();
                for (int i = 0; i < TrainingData.size(); i++) {
                    Double[] SingleEntry = TrainingData.get(i);
                    double[][] InputVec = new double[InitialANN.getInputNum()][1];
                    for (int k = 0; k < SingleEntry.length - 1; k++) {
                        InputVec[k][0] = SingleEntry[k];
                    }
                    double EntryTag = SingleEntry[SingleEntry.length - 1];
                    ForwardResult = FinalANN.getForwardResult(InputVec);
                    ErrVec[0][0] = EntryTag - ForwardResult[0][0];
                    SquareErrSum += Math.pow(ErrVec[0][0], 2);
                    ErrorArr[i][0] = ErrVec[0][0];
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
                if (SquareErrSum / TrainingData.size() >= MSE_UpBound) {
                    ALPHA = GAMMA / (2 * WeightSquareSum);
                    BETA = (TrainingData.size() - GAMMA) / (2 * SquareErrSum);
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

                    Zmat NetworkUpdates = Times.o(new Z(-1, 0), Times.o(Times.o(Inv.o(G), transpose.o(TotalJacobianMatrix)), new Zmat(ErrorArr)));
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
                    SumUpdatesANN.updateWeightNetwork(UpdatesANN.getANN());
                }
                else{
                    SumUpdatesANN.MSEofCertainSet=SquareErrSum / TrainingData.size();
                }
            }
        } catch (Exception e) {

        }finally {
            return SumUpdatesANN;
        }

    }
}
