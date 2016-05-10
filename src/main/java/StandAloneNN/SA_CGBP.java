package StandAloneNN;

import FileIO.FileReadNWrite;
import Jampack.JampackException;
import NeuralNetwork.ArtificialNeuralNetwork;
import NeuralNetwork.LinearSearchMinumum;
import NeuralNetwork.NeuronLayer;

import javax.swing.text.html.HTML;
import java.io.IOException;
import java.util.HashSet;
import java.util.Vector;

import static FileIO.FileReadNWrite.readTxtFile;

/**
 * Created by mlx on 4/9/16.
 */
public class SA_CGBP {
    public static void main(String[] args) throws JampackException, IOException {
        Vector<Double[]> InputPair = readTxtFile("/home/mlx/Documents/DataSet/StandAloneFuncTrainData_WithNoise");
        Vector<Double[]> TestPair = readTxtFile("/home/mlx/Documents/DataSet/StandAloneFuncTestData");

        double MSE_Bound = 0.001;
        double LeastGradientLength = 1E-5;
        double HeuristicStep = 0.1;

        int InputNum = 1;
        int LayerNum = 3;
        int[] NumEachLayer = {5, 7, 1};
        int[] IndexEachLayer = {1, 1, 3};

        ArtificialNeuralNetwork FinalANN = new ArtificialNeuralNetwork(InputNum, LayerNum, NumEachLayer, IndexEachLayer);
        ArtificialNeuralNetwork BatchStore = new ArtificialNeuralNetwork(FinalANN.getANN());
//        ArtificialNeuralNetwork GradientStore = new ArtificialNeuralNetwork(FinalANN.getANN());

        int ParaNum = 0;
        for (int i = 0; i < FinalANN.getLayerNum(); i++) {
            ParaNum += FinalANN.getANN()[i].getNeuronNum() * (FinalANN.getANN()[i].getInputNum() + 1);
        }

        double[][] ForwardResult = null;
        double[][] ErrVec = new double[FinalANN.getOutputNum()][1];
        NeuronLayer[] LastGenerationGradient = null;
        NeuronLayer[] LastGenerationDirection = null;

//        int PackageSize = 100;
//        Vector<Vector<Double[]>> InputPairPackage = new Vector<Vector<Double[]>>();
//        for (int i = 0; i < InputPair.size(); ) {
//            Vector<Double[]> OnePackInput = new Vector<Double[]>();
//            for (int j = 0; j < PackageSize && i < InputPair.size(); j++, i++) {
//                int Random=getRandomNum(0,InputPair.size()-1);
//                while (!h.contains(Random)){
//                    Random=getRandomNum(0,InputPair.size()-1);
//                    h.add(Random);
//                }
//                OnePackInput.add(InputPair.get(Random));
//            }
//            InputPairPackage.add(OnePackInput);
//        }
        System.out.println("IterationNum\tMSE\tGradientLength");
        for (int time = 0; time < 2000; time++) {
            double TotalMSE = 0;
            BatchStore.clearNetwork();
            for (int p = 0; p < InputPair.size(); p++) {
                Double[] OneEntry = InputPair.get(p);
                double Tag = OneEntry[OneEntry.length - 1];
                double[][] InputVec = new double[InputNum][1];
                for (int k = 0; k < OneEntry.length - 1; k++) {
                    InputVec[k][0] = OneEntry[k];
                }
                ForwardResult = FinalANN.getForwardResult(InputVec);
                ErrVec[0][0] = Tag - ForwardResult[0][0];
                TotalMSE += Math.pow(ErrVec[0][0], 2);
                BatchStore.updateWeightNetwork(FinalANN.getErrorGradient(ErrVec));
            }

            BatchStore.averageNetwork(InputPair.size());
            NeuronLayer[] ThisGenerationGradient = BatchStore.getANN();
            NeuronLayer[] UpdatesDirection = null;
            NeuronLayer[] SearchDirection = null;

            double LengthOfGradient = ArtificialNeuralNetwork.getNeuronLayerArrLength_norm2(ThisGenerationGradient);
            System.out.println(String.valueOf(time) + "\t" + String.valueOf(TotalMSE / InputPair.size()) + "\t" + String.valueOf(LengthOfGradient));

            if ((TotalMSE / InputPair.size() < MSE_Bound) || (LengthOfGradient < LeastGradientLength)) {
                break;
            }

            if (time % ParaNum == 0) {
                SearchDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(ThisGenerationGradient, -1.0);
            } else {
                SearchDirection = ArtificialNeuralNetwork.getFR_CGD(ThisGenerationGradient, LastGenerationGradient, LastGenerationDirection);
            }
            double[] IntervalLocation = LinearSearchMinumum.getIntervalLocation(FinalANN, InputPair, SearchDirection, HeuristicStep);
            double OptimumLearningRate = LinearSearchMinumum.getTolerableMinimum(FinalANN, InputPair, SearchDirection, IntervalLocation, 0.02);
            UpdatesDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, OptimumLearningRate);
            FinalANN.updateWeightNetwork(UpdatesDirection);
            LastGenerationGradient = ThisGenerationGradient;
            LastGenerationDirection = SearchDirection;
        }

        double[][] TestForwardResult;
        double TestErrSum = 0.0;
        for (int i = 0; i < TestPair.size(); i++) {
            Double[] temp = (TestPair.get(i));
            double[][] InputVec = new double[InputNum][1];
            for (int k = 0; k < temp.length - 1; k++) {
                InputVec[k][0] = temp[k];
            }
            double Tag = temp[temp.length - 1];

            TestForwardResult = FinalANN.getForwardResult(InputVec);
            System.out.println(TestForwardResult[0][0]);
            TestErrSum += (Tag - TestForwardResult[0][0]) * (Tag - TestForwardResult[0][0]);
        }
        TestErrSum /= TestPair.size();
        System.out.println("\n" + String.valueOf(TestErrSum));

        String[] SaveANN = FinalANN.saveANN();
        FileReadNWrite.LocalWriteFile("/home/mlx/Documents/ResultANN/CGBP_FinalANN", SaveANN);

    }
    public static int getRandomNum(int m,int n) {
        return (m + (int) (Math.random() * (n - m + 1)));
    }
}
