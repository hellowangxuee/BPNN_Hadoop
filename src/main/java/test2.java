import FileIO.FileReadNWrite;
import HDFS_IO.ReadNWrite;
import Jampack.*;
import MapReduce.BPTrain;
import NeuralNetwork.ArtificialNeuralNetwork;
import NeuralNetwork.LinearSearchMinumum;
import NeuralNetwork.NeuronLayer;
import org.apache.commons.configuration.SystemConfiguration;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec;
import org.mortbay.jetty.HttpParser;

import javax.swing.text.html.HTML;
import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Vector;

import static FileIO.FileReadNWrite.readTxtFile;

/**
 * Created by mlx on 3/18/16.
 */
public class test2 {
    public static void main(String[] args) throws JampackException, IOException {
        Vector<Double[]> InputPair = readTxtFile("/home/mlx/Documents/DataSet/DIS_TEP_TestData");
        System.out.println(InputPair.size());
        int InputNum = 41;
        double[][] InputVec = new double[InputNum][1];
        double[][] ForwardResult = null;
        Double[] temp = null;

        ArtificialNeuralNetwork fANN = new ArtificialNeuralNetwork("hdfs://Master:9000/TEP_Classify/TEP_moVLBP-v2/result_ANN-22");
        double[] PreResult = new double[InputPair.size()];
        double[] TagArr = new double[InputPair.size()];
        for (int i = 0; i < InputPair.size(); i++) {
            temp = (InputPair.get(i));
            double Tag = temp[temp.length - 1];
            for (int k = 0; k < temp.length - 1; k++) {
                InputVec[k][0] = temp[k];
            }
            ForwardResult = fANN.getForwardResult(InputVec);
            PreResult[i] = ForwardResult[0][0];
            TagArr[i] = Tag;
            fANN.clearTempResult();
        }
        for (double Threshold = 0.0; Threshold < 1.1; Threshold += 0.01) {
            int RightNum = 0;
            int WrongNum = 0;
            for (int k = 0; k < InputPair.size(); k++) {
                if ((PreResult[k] >= Threshold && TagArr[k] == 1.0) || (PreResult[k] < Threshold && TagArr[k] == 0.0)) {
                    RightNum++;
                } else {
                    WrongNum++;
                }
            }
            double Accuracy = (double) (RightNum) / (RightNum + WrongNum);
            System.out.println(String.valueOf(Threshold) + "\t" + String.valueOf(Accuracy));
        }
//
////        double Accuracy=getAccuracy(fANN,"/home/mlx/Documents/TestingDataset",0.05);
////        System.out.println(String.valueOf(0.05)+"\t"+String.valueOf(Accuracy));
////        String ANNpath="hdfs://Master:9000/user/mlx/FinalANN-v1.3";
////        ArtificialNeuralNetwork fANN=new ArtificialNeuralNetwork(ANNpath);
////        String[] ann=fANN.saveANN();
////        ReadNWrite.hdfs_Write(ann,"hdfs://Master:9000/user/mlx/FinalANN-v1.4");
//
////        String PathPrefix="hdfs://Master:9000/user/mlx/Simu_EXP5/VLBP_update-";
////        for (int i=0;i<50;i++){
////            String path=PathPrefix+String.valueOf(i);
////            Map<String,Double> map= BPTrain.getWeightUpdatesFromFile(path);
////            System.out.println(String.valueOf(i)+"\t"+String.valueOf(map.get("TotalSquareError")) );
////        }
//
//        Vector InputPair = readTxtFile("/home/mlx/Documents/TrainData");
//        Vector TestPair = readTxtFile("/home/mlx/Documents/TestData");
//        HashSet h = new HashSet();
//
//        int InputNum1 = 1;
//        int LayerNum1 = 2;
//        int[] NumEachLayer = {3, 1};
//        int[] IndexEachLayer = {1, 3};
//
//        ArtificialNeuralNetwork TrainingANN = new ArtificialNeuralNetwork(InputNum1, LayerNum1, NumEachLayer, IndexEachLayer);
//        ArtificialNeuralNetwork BatchStoreWork = new ArtificialNeuralNetwork(InputNum1, LayerNum1, NumEachLayer, IndexEachLayer);
//        BatchStoreWork.clearNetwork();
//
//        double[][] InputVec = new double[InputNum1][1];
//        double[][] ErrVec = new double[1][1];
//        double[][] ForwardResult;
//        double ThisErrSum = 0.0;
//        double LastErrSum= 0.0;
//        double[] temp;
//        double[][] ErrArr = new double[InputPair.size()][1];
////        int ParaNum = 0;
////        for (int i = 0; i < LayerNum; i++) {
////            ParaNum += TrainingANN.getANN()[i].getNeuronNum() * (TrainingANN.getANN()[i].getInputNum() + 1);
////        }
////
////        double HeuristicStep = 0.1;
////
////        NeuronLayer[] LastGenerationGradient = null;
////        NeuronLayer[] LastGenerationDirection = null;
////
//        Zmat TotalJacobianMatrix=null;
//
//        for (int i = 0; i < InputPair.size(); i++) {
//            temp = (double[]) (InputPair.get(i));
//            for (int k = 0; k < temp.length - 1; k++) {
//                InputVec[k][0] = temp[k];
//            }
//            ForwardResult = TrainingANN.getForwardResult(InputVec);
//            ErrVec = new double[TrainingANN.getOutputNum()][1];
//            double Tag = temp[temp.length - 1];
//            ErrVec[0][0] = Tag - ForwardResult[0][0];
//            ThisErrSum += ErrVec[0][0] * ErrVec[0][0];
//            ErrArr[i][0] = ErrVec[0][0];
////                for(int k=0;Math.abs(ErrVec[0][0]) >=0.001;k++) {
////                    NeuronLayer[] ThisGenerationGradient = TrainingANN.getErrorGradient(ErrVec);
////
////                    double GradientLength=0;
////                    for(int p=0;p<ThisGenerationGradient.length;p++) {
////                        for(int l=0;l<ThisGenerationGradient[p].getNeuronNum();l++){
////                            for(int q=0;q<ThisGenerationGradient[p].getInputNum();q++){
////                                GradientLength+= ThisGenerationGradient[p].getCertainWeight(l,q) *ThisGenerationGradient[p].getCertainWeight(l,q);
////                            }
////                            GradientLength+= ThisGenerationGradient[p].getCertainBias(l) *ThisGenerationGradient[p].getCertainBias(l);
////                        }
////                    }
////                    GradientLength=Math.sqrt(GradientLength);
////                    if(GradientLength< 0.05){
////                        break;
////                    }
////
////                    NeuronLayer[] UpdatesDirection;
////                    NeuronLayer[] SearchDirection;
////                    if (k % ParaNum == 0) {
////                        SearchDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(ThisGenerationGradient, -1.0);
////                    } else {
////                        SearchDirection = ArtificialNeuralNetwork.getFR_CGD(ThisGenerationGradient, LastGenerationGradient, LastGenerationDirection);
////                    }
////                    double[] IntervalLocation = LinearSearchMinumum.getIntervalLocation(TrainingANN, InputVec, Tag, SearchDirection, HeuristicStep);
////                    double OptimumLearningRate = LinearSearchMinumum.getTolerableMinimum(TrainingANN, InputVec, Tag, SearchDirection, IntervalLocation, 0.02);
////                    UpdatesDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, OptimumLearningRate);
////
////                    LastGenerationGradient = ThisGenerationGradient;
////                    LastGenerationDirection = SearchDirection;
////                    TrainingANN.updateWeightNetwork(UpdatesDirection);
////                    BatchStoreWork.updateWeightNetwork(UpdatesDirection);
////                    ForwardResult = TrainingANN.getForwardResult(InputVec);
////                    ErrVec[0][0] = Tag - ForwardResult[0][0];
////                }
////
////                RandomNum = test.getRandomNum(0, InputPair.size() - 1);
////                h.add(RandomNum);
////            }
////            h.clear();
////            finalANN.updateWeightNetwork(BatchStoreWork);
////            BatchStoreWork.clearNetwork();
////            System.out.println(String.valueOf(t)+"\t"+String.valueOf(ErrSum));
////
////        }
////        //System.out.println(String.valueOf(ErrSum));
////        double TestErrSum = 0.0;
////        for (int i = 0; i < TestPair.size(); i++) {
////            temp = (double[]) (TestPair.get(i));
////            for (int k = 0; k < temp.length - 1; k++) {
////                InputVec[k][0] = temp[k];
////            }
////            double[][] ForwardResult1 = finalANN.getForwardResult(InputVec);
////            double Tag = temp[temp.length - 1];
////            ErrVec[0][0] = Tag - ForwardResult1[0][0];
////            TestErrSum += ErrVec[0][0] * ErrVec[0][0];
////            System.out.println(String.valueOf(ForwardResult1[0][0]));
////        }
////        System.out.println();
////        //TestErrSum/=TestPair.size();
////        System.out.println(TestErrSum);
//            if (i == 0) {
//                TotalJacobianMatrix = TrainingANN.getJacobianMatrix();
//            } else {
//                try {
//                    TotalJacobianMatrix = Merge.o21(TotalJacobianMatrix, TrainingANN.getJacobianMatrix());
//                } catch (Exception e) {
//                    System.out.println(e.toString());
//                    continue;
//                }
//            }
//        }
////        MSE = ThisTimeSquErrSum / LineDataArr.length;
//        LastErrSum = ThisErrSum;
//        //System.out.println("-1\t"+String.valueOf(ThisErrSum));
//        double miu=0.01;
//        double MultipliedFactor=10;
//        for(int t=0;t<50;t++){
//            try {
//                System.out.println(String.valueOf(t)+"\t"+String.valueOf(ThisErrSum));
//                Zmat H = Times.o(transpose.o(TotalJacobianMatrix), TotalJacobianMatrix);
//                int OrderOfH = H.nr;
//                Eig EigOfH = new Eig(H);
//                double minEigValue = EigOfH.D.get0(OrderOfH-1).re;
//                if (minEigValue < 0 && miu + minEigValue < 0) {
//                    miu += (-minEigValue);
//                }
//                Zmat UnitMatrix=Eye.o(OrderOfH);
//                Zmat G = Plus.o(H, Times.o(new Z(miu, 0), UnitMatrix));
//                Zmat NetworkUpdates = Times.o(new Z(-1, 0), Times.o(Times.o(Inv.o(G), transpose.o(TotalJacobianMatrix)), new Zmat(ErrArr)));
//                ArtificialNeuralNetwork UpdatesANN = new ArtificialNeuralNetwork(TrainingANN.getANN());
//
//                int LayerNum = 0;
//                int NeuronNum = 0;
//                int InputNum = 0;
//                for (int index = 0; index < NetworkUpdates.nr; index++) {
//                    int LayerParaSum = 0;
//                    for (int i = 0; i < UpdatesANN.getLayerNum(); i++) {
//                        LayerParaSum += UpdatesANN.getANN()[i].getNeuronNum() * (UpdatesANN.getANN()[i].getInputNum() + 1);
//                        if (index / LayerParaSum < 1) {
//                            LayerNum = i;
//                            break;
//                        }
//                    }
//                    int NeuronOffset = index - LayerParaSum + UpdatesANN.getANN()[LayerNum].getNeuronNum() * (UpdatesANN.getANN()[LayerNum].getInputNum() + 1);
//                    if (NeuronOffset >= (UpdatesANN.getANN()[LayerNum].getNeuronNum() * UpdatesANN.getANN()[LayerNum].getInputNum())) {
//                        int BiasNum = NeuronOffset - (UpdatesANN.getANN()[LayerNum].getNeuronNum() * UpdatesANN.getANN()[LayerNum].getInputNum());
//                        UpdatesANN.setCertainBias(LayerNum, BiasNum, NetworkUpdates.get0(index, 0).re);
//                    } else {
//                        NeuronNum = NeuronOffset / UpdatesANN.getANN()[LayerNum].getInputNum();
//                        InputNum = NeuronOffset % UpdatesANN.getANN()[LayerNum].getInputNum();
//                        UpdatesANN.setCertainWeight(LayerNum, NeuronNum, InputNum, NetworkUpdates.get0(index, 0).re);
//                    }
//                }
//                TrainingANN.updateWeightNetwork(UpdatesANN.getANN());
//                Zmat TempJacobian=null;
//                ThisErrSum = 0;
//                for (int EntryNum = 0; EntryNum < InputPair.size(); EntryNum++) {
//                    temp = (double[]) (InputPair.get(EntryNum));
//                    for (int k = 0; k < temp.length - 1; k++) {
//                        InputVec[k][0] = temp[k];
//                    }
//                    ForwardResult = TrainingANN.getForwardResult(InputVec);
//                    ErrVec = new double[TrainingANN.getOutputNum()][1];
//                    double Tag = temp[temp.length - 1];
//                    ErrVec[0][0] = Tag - ForwardResult[0][0];
//                    ThisErrSum += ErrVec[0][0] * ErrVec[0][0];
//                    ErrArr[EntryNum][0] = ErrVec[0][0];
//                    if (EntryNum == 0) {
//                        TempJacobian = TrainingANN.getJacobianMatrix();
//                    } else {
//                        TempJacobian = Merge.o21(TempJacobian, TrainingANN.getJacobianMatrix());
//                    }
//                }
//                if (ThisErrSum < LastErrSum) {
//                    miu /= MultipliedFactor;
//                    TotalJacobianMatrix=new Zmat(TempJacobian) ;
//                    BatchStoreWork.updateWeightNetwork(UpdatesANN.getANN());
//                    LastErrSum = ThisErrSum;
//                } else {
//                    miu *= MultipliedFactor;
//                    TrainingANN.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(UpdatesANN.getANN(),-1));
//                }
//                double MSE = ThisErrSum / InputPair.size();
//                if(MSE<0.001){
//                    System.out.println(String.valueOf(t+1)+"\t"+String.valueOf(ThisErrSum));
//                    break;
//                }
//
//
//            } catch (Exception e) {
//                System.out.println(e.toString());
//
//                break;
//            }
//        }
//        Vector<Double[]> InputPair = readTxtFile("/home/mlx/Documents/DataSet/DIS_TEP_TrainData_part2");
//        String[] n=new String[1];
//        for(int i=0;i<InputPair.size();i++) {
//            String TotalInput="";
//            Double[] temp = (InputPair.get(i));
//            for (int j = 0; j < temp.length; j++) {
//                if (j != temp.length - 1) {
//                    TotalInput += String.valueOf(temp[j]) + "\t";
//                } else {
//                    TotalInput += String.valueOf(temp[j]);
//                }
//            }
//            TotalInput += ";";
//            n[0]=TotalInput;
//            //int Random=getRandomNum(9,16);
//            FileReadNWrite.LocalWriteFile_NoNewLine("/home/mlx/Documents/TEP_DataSet/DISTEP_TrainingData_part2",n);
    }

    public static double getAccuracy(ArtificialNeuralNetwork testANN, String datasetPath, double Threshold) {
        Vector dataArr = readTxtFile(datasetPath);
        double[] temp = (double[]) dataArr.get(0);
        double[][] InputVec = new double[temp.length - 1][1];
        int RightNum = 0;
        int WrongNum = 0;
        for (int i = 0; i < dataArr.size(); i++) {
            temp = (double[]) (dataArr.get(i));
            for (int k = 0; k < temp.length - 1; k++) {
                InputVec[k][0] = temp[k];
            }
            double[][] ForwardResult = testANN.getForwardResult(InputVec);
            int TrueFlag = (int) (temp[temp.length - 1]);
            int PreFlag = 0;
            if (ForwardResult[0][0] >= Threshold) {
                PreFlag = 1;
            }
            if (TrueFlag == PreFlag) {
                RightNum += 1;
            } else {
                WrongNum += 1;
            }
        }
        double Accuracy = (double) (RightNum) / (RightNum + WrongNum);
        return Accuracy;
    }
    public static int getRandomNum(int m,int n) {
        return (m + (int) (Math.random() * (n - m + 1)));
    }
}
