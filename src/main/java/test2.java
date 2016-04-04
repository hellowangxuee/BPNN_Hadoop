import FileIO.FileReadNWrite;
import HDFS_IO.ReadNWrite;
import Jampack.JampackException;
import MapReduce.BPTrain;
import NeuralNetwork.ArtificialNeuralNetwork;
import NeuralNetwork.LinearSearchMinumum;
import NeuralNetwork.NeuronLayer;
import org.apache.commons.configuration.SystemConfiguration;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec;

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

//        for(double Threshold=0.001;Threshold<0.5;Threshold+=0.001) {
////            int NormalRightNum=0;
////            int NormalWrongNum=0;
////            int Fault1RightNum=0;
////            int Fault1WrongNum=0;
//            int RightNum=0;
//            int WrongNum=0;
//            for (int i = 0; i < InputPair.size(); i++) {
//                temp = (double[]) (InputPair.get(i));
//                for (int k = 0; k < temp.length - 1; k++) {
//                    InputVec[k][0] = temp[k];
//                }
//                ForwardResult = fANN.getForwardResult(InputVec);
//                int TrueFlag=(int)(temp[temp.length-1]);
//                int PreFlag=0;
//                if(ForwardResult[0][0]>=Threshold){
//                    PreFlag=1;
//                }
//                if(TrueFlag==PreFlag){
//                    RightNum+=1;
//                }
//                else{
//                    WrongNum+=1;
//                }
////                System.out.println(String.valueOf(i) + "\t" + String.valueOf(temp[temp.length - 1]) + "\t" + String.valueOf(ForwardResult[0][0]));
//            }
//            double Accuracy=(double)(RightNum)/(RightNum+WrongNum);
//            System.out.println(String.valueOf(Threshold)+"\t"+String.valueOf(Accuracy));
//        }

//        double Accuracy=getAccuracy(fANN,"/home/mlx/Documents/TestingDataset",0.05);
//        System.out.println(String.valueOf(0.05)+"\t"+String.valueOf(Accuracy));
//        String ANNpath="hdfs://Master:9000/user/mlx/FinalANN-v1.3";
//        ArtificialNeuralNetwork fANN=new ArtificialNeuralNetwork(ANNpath);
//        String[] ann=fANN.saveANN();
//        ReadNWrite.hdfs_Write(ann,"hdfs://Master:9000/user/mlx/FinalANN-v1.4");

//        String PathPrefix="hdfs://Master:9000/user/mlx/Simu_EXP5/VLBP_update-";
//        for (int i=0;i<50;i++){
//            String path=PathPrefix+String.valueOf(i);
//            Map<String,Double> map= BPTrain.getWeightUpdatesFromFile(path);
//            System.out.println(String.valueOf(i)+"\t"+String.valueOf(map.get("TotalSquareError")) );
//        }

        Vector InputPair = readTxtFile("/home/mlx/Documents/TrainData");
        Vector TestPair = readTxtFile("/home/mlx/Documents/TestData");
        HashSet h = new HashSet();

        int InputNum = 1;
        int LayerNum = 2;
        int[] NumEachLayer = {2, 1};
        int[] IndexEachLayer = {1, 3};

        ArtificialNeuralNetwork TrainingANN = new ArtificialNeuralNetwork(InputNum, LayerNum, NumEachLayer, IndexEachLayer);
        //ArtificialNeuralNetwork TrainingANN = new ArtificialNeuralNetwork("hdfs://Master:9000/user/mlx/EX_ANN");
        ArtificialNeuralNetwork finalANN=new ArtificialNeuralNetwork(TrainingANN.getANN());
        ArtificialNeuralNetwork BatchStoreWork = new ArtificialNeuralNetwork(InputNum, LayerNum, NumEachLayer, IndexEachLayer);
        BatchStoreWork.clearNetwork();

        double[][] InputVec = new double[InputNum][1];
        double[][] ErrVec = new double[1][1];
        double[][] ForwardResult;
        double ErrSum = 0.0;
        double[] temp;
        int ParaNum = 0;
        for (int i = 0; i < LayerNum; i++) {
            ParaNum += TrainingANN.getANN()[i].getNeuronNum() * (TrainingANN.getANN()[i].getInputNum() + 1);
        }

        double HeuristicStep = 0.1;

        NeuronLayer[] LastGenerationGradient = null;
        NeuronLayer[] LastGenerationDirection = null;

        for (int t = 0; t < 15; t++) {
            ErrSum = 0.0;
            int RandomNum = test.getRandomNum(0, InputPair.size() - 1);
            h.add(RandomNum);
            for (int i = 0; i < InputPair.size(); i++) {
                while (!h.contains(RandomNum)) {
                    RandomNum = test.getRandomNum(0, InputPair.size() - 1);
                    h.add(RandomNum);
                }
                temp = (double[]) (InputPair.get(RandomNum));
                for (int k = 0; k < temp.length - 1; k++) {
                    InputVec[k][0] = temp[k];
                }
                ForwardResult = TrainingANN.getForwardResult(InputVec);
                ErrVec = new double[TrainingANN.getOutputNum()][1];
                double Tag = temp[temp.length - 1];
                ErrVec[0][0] = Tag - ForwardResult[0][0];
                ErrSum += ErrVec[0][0] * ErrVec[0][0];
                for(int k=0;Math.abs(ErrVec[0][0]) >=0.001;k++) {
                    NeuronLayer[] ThisGenerationGradient = TrainingANN.getErrorGradient(ErrVec);

                    double GradientLength=0;
                    for(int p=0;p<ThisGenerationGradient.length;p++) {
                        for(int l=0;l<ThisGenerationGradient[p].getNeuronNum();l++){
                            for(int q=0;q<ThisGenerationGradient[p].getInputNum();q++){
                                GradientLength+= ThisGenerationGradient[p].getCertainWeight(l,q) *ThisGenerationGradient[p].getCertainWeight(l,q);
                            }
                            GradientLength+= ThisGenerationGradient[p].getCertainBias(l) *ThisGenerationGradient[p].getCertainBias(l);
                        }
                    }
                    GradientLength=Math.sqrt(GradientLength);
                    if(GradientLength< 0.05){
                        break;
                    }

                    NeuronLayer[] UpdatesDirection;
                    NeuronLayer[] SearchDirection;
                    if (k % ParaNum == 0) {
                        SearchDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(ThisGenerationGradient, -1.0);
                    } else {
                        SearchDirection = ArtificialNeuralNetwork.getFR_CGD(ThisGenerationGradient, LastGenerationGradient, LastGenerationDirection);
                    }
                    double[] IntervalLocation = LinearSearchMinumum.getIntervalLocation(TrainingANN, InputVec, Tag, SearchDirection, HeuristicStep);
                    double OptimumLearningRate = LinearSearchMinumum.getTolerableMinimum(TrainingANN, InputVec, Tag, SearchDirection, IntervalLocation, 0.02);
                    UpdatesDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, OptimumLearningRate);

                    LastGenerationGradient = ThisGenerationGradient;
                    LastGenerationDirection = SearchDirection;
                    TrainingANN.updateWeightNetwork(UpdatesDirection);
                    BatchStoreWork.updateWeightNetwork(UpdatesDirection);
                    ForwardResult = TrainingANN.getForwardResult(InputVec);
                    ErrVec[0][0] = Tag - ForwardResult[0][0];
                }

                RandomNum = test.getRandomNum(0, InputPair.size() - 1);
                h.add(RandomNum);
            }
            h.clear();
            finalANN.updateWeightNetwork(BatchStoreWork);
            BatchStoreWork.clearNetwork();
            System.out.println(String.valueOf(t)+"\t"+String.valueOf(ErrSum));

        }
        //System.out.println(String.valueOf(ErrSum));
        double TestErrSum = 0.0;
        for (int i = 0; i < TestPair.size(); i++) {
            temp = (double[]) (TestPair.get(i));
            for (int k = 0; k < temp.length - 1; k++) {
                InputVec[k][0] = temp[k];
            }
            double[][] ForwardResult1 = finalANN.getForwardResult(InputVec);
            double Tag = temp[temp.length - 1];
            ErrVec[0][0] = Tag - ForwardResult1[0][0];
            TestErrSum += ErrVec[0][0] * ErrVec[0][0];
            System.out.println(String.valueOf(ForwardResult1[0][0]));
        }
        System.out.println();
        //TestErrSum/=TestPair.size();
        System.out.println(TestErrSum);

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
}
