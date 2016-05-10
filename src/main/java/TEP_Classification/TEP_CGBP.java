package TEP_Classification;

import FileIO.FileReadNWrite;
import Jampack.JampackException;
import NeuralNetwork.ArtificialNeuralNetwork;
import NeuralNetwork.LinearSearchMinumum;
import NeuralNetwork.NeuronLayer;
import org.apache.hadoop.hdfs.web.resources.DeleteOpParam;
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec;
import sun.security.util.Length;

import java.io.IOException;
import java.util.Vector;

import static FileIO.FileReadNWrite.readTxtFile;

/**
 * Created by mlx on 5/9/16.
 */
public class TEP_CGBP {
    public static void main(String[] args) throws JampackException, IOException {
        Vector<Double[]> TrainData = readTxtFile("/home/mlx/Documents/DataSet/TEP_TrainData");
        Vector<Double[]> TestData = readTxtFile("/home/mlx/Documents/DataSet/TEP_TestData");

        double MSE_Bound = 0.05;
        double LeastGradientLength = 1E-5;
        double HeuristicStep = 0.1;

        int InputNum = 41;
        int LayerNum = 3;
        int[] NumEachLayer = {7, 10, 1};
        int[] IndexEachLayer = {4, 4, 4};

        ArtificialNeuralNetwork FinalANN = new ArtificialNeuralNetwork(InputNum, LayerNum, NumEachLayer, IndexEachLayer);
        ArtificialNeuralNetwork BatchStore = new ArtificialNeuralNetwork(FinalANN.getANN());

        int ParaNum = 0;
        for (int i = 0; i < FinalANN.getLayerNum(); i++) {
            ParaNum += FinalANN.getANN()[i].getNeuronNum() * (FinalANN.getANN()[i].getInputNum() + 1);
        }

        double[][] ForwardResult = null;
        double[][] ErrVec = new double[FinalANN.getOutputNum()][1];
        NeuronLayer[] LastGenerationGradient = null;
        NeuronLayer[] LastGenerationDirection = null;
        NeuronLayer[] UpdatesDirection = null;
        NeuronLayer[] SearchDirection = null;
//        int PackageSize = 100;
//        Vector<Vector<Double[]>> TrainDataPackage = new Vector<Vector<Double[]>>();
//        for (int i = 0; i < TrainData.size(); ) {
//            Vector<Double[]> OnePackInput = new Vector<Double[]>();
//            for (int j = 0; j < PackageSize && i < TrainData.size(); j++, i++) {
//                int Random=getRandomNum(0,TrainData.size()-1);
//                while (!h.contains(Random)){
//                    Random=getRandomNum(0,TrainData.size()-1);
//                    h.add(Random);
//                }
//                OnePackInput.add(TrainData.get(Random));
//            }
//            TrainDataPackage.add(OnePackInput);
//        }
        System.out.println("IterationNum\tMSE\tGradientLength");
        String[] RunningMessage = new String[10000];
        for (int time = 0; time < 2000; time++) {
            double TotalMSE = 0;
            BatchStore.clearNetwork();
            for (int p = 0; p < TrainData.size(); p++) {
                Double[] OneEntry = TrainData.get(p);
                double Tag = OneEntry[OneEntry.length - 1];
                double[][] InputVec = new double[InputNum][1];
                for (int k = 0; k < OneEntry.length - 1; k++) {
                    InputVec[k][0] = OneEntry[k];
                }
                ForwardResult = FinalANN.getForwardResult(InputVec);
                ErrVec[0][0] = Tag - ForwardResult[0][0];
                TotalMSE += Math.pow(ErrVec[0][0], 2);
                NeuronLayer[] GradientOfEntry=FinalANN.getErrorGradient(ErrVec);
                if(String.valueOf(ArtificialNeuralNetwork.getNeuronLayerArrLength_norm2(GradientOfEntry)).equals("NaN")){
                    continue;
                }
                BatchStore.updateWeightNetwork(GradientOfEntry);
            }
//            if (TotalMSE == Double.NaN) {
//                FinalANN.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(UpdatesDirection, -1));
//                break;
//            }
            BatchStore.averageNetwork(TrainData.size());
            NeuronLayer[] ThisGenerationGradient = BatchStore.getANN();
            double LengthOfGradient = ArtificialNeuralNetwork.getNeuronLayerArrLength_norm2(ThisGenerationGradient);
            if(String.valueOf(LengthOfGradient).equals("NaN")){
                FinalANN.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(UpdatesDirection,-1.0));
                break;
            }

            if ((TotalMSE / TrainData.size() < MSE_Bound) || (LengthOfGradient < LeastGradientLength)) {
                break;
            }

            if (time % ParaNum == 0) {
                SearchDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(ThisGenerationGradient, -1.0);
            } else {
                int random = getRandomNum(0, 1);
                if (random == 0) {
                    SearchDirection = ArtificialNeuralNetwork.getPR_CGD(ThisGenerationGradient, LastGenerationGradient, LastGenerationDirection);
                } else {
                    SearchDirection = ArtificialNeuralNetwork.getFR_CGD(ThisGenerationGradient, LastGenerationGradient, LastGenerationDirection);
                }
            }
            double[] IntervalLocation = LinearSearchMinumum.getIntervalLocation(FinalANN, TrainData, SearchDirection, HeuristicStep);
            double OptimumLearningRate = LinearSearchMinumum.getTolerableMinimum(FinalANN, TrainData, SearchDirection, IntervalLocation, 0.02);
            if(OptimumLearningRate>5){
                OptimumLearningRate*=0.1;
            }
            UpdatesDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, OptimumLearningRate);
            double UpdateLength=ArtificialNeuralNetwork.getNeuronLayerArrLength_norm2(UpdatesDirection);

            System.out.println(String.valueOf(time) + "\t" +TotalMSE+"\t"+ String.valueOf(TotalMSE / TrainData.size()) + "\t" + String.valueOf(LengthOfGradient)+"\t"+String.valueOf(UpdateLength)+"\t"+OptimumLearningRate);
            RunningMessage[time] = String.valueOf(time) + "\t" + String.valueOf(TotalMSE / TrainData.size()) + "\t" + String.valueOf(LengthOfGradient)+"\t"+String.valueOf(UpdateLength)+"\t"+String.valueOf(OptimumLearningRate);
//            if(ArtificialNeuralNetwork.getNeuronLayerArrLength_norm_max(UpdatesDirection)>10){
//                UpdatesDirection= ArtificialNeuralNetwork.multiplyNeuronLayers(UpdatesDirection,0.001);
//            }
            FinalANN.updateWeightNetwork(UpdatesDirection);
            LastGenerationGradient = ThisGenerationGradient;
            LastGenerationDirection = SearchDirection;
        }
        FileReadNWrite.LocalWriteFile("/home/mlx/Documents/ProcessResult/TEP_CGBP_IteMsg-v2", RunningMessage);

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
        FileReadNWrite.LocalWriteFile("/home/mlx/Documents/ResultANN/TEP_CGBP_FinalANN-v2", SaveANN);
        FileReadNWrite.LocalWriteFile("/home/mlx/Documents/ProcessResult/TEP_CGBP_PreRe-v2", TestResult);

    }

    public static int getRandomNum(int m,int n) {
        return (m + (int) (Math.random() * (n - m + 1)));
    }
}

