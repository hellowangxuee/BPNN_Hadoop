package TEP_Classification;

import FileIO.FileReadNWrite;
import Jampack.JampackException;
import NeuralNetwork.ArtificialNeuralNetwork;

import java.io.IOException;
import java.util.Vector;

import static FileIO.FileReadNWrite.readTxtFile;

/**
 * Created by mlx on 5/9/16.
 */
public class TEP_SDBP {
    public static void main(String[] args) throws JampackException, IOException {
        Vector<Double[]> TrainData = readTxtFile("/home/mlx/Documents/DataSet/TEP_TrainData");
        Vector<Double[]> TestData = readTxtFile("/home/mlx/Documents/DataSet/TEP_TestData");

        int InputNum = 41;
        int LayerNum = 3;
        int[] NumEachLayer = {7, 10, 1};
        int[] IndexEachLayer = {1, 4, 4};
        double MSE_upbound = 0.01;

        double[][] ErrVec = new double[NumEachLayer[NumEachLayer.length - 1]][1];
        double[][] ForwardResult;

        ArtificialNeuralNetwork FinalANN = new ArtificialNeuralNetwork(InputNum, LayerNum, NumEachLayer, IndexEachLayer);
        ArtificialNeuralNetwork BatchStore = new ArtificialNeuralNetwork(FinalANN.getANN());

        String[] RunningMessage = new String[10000];

        System.out.println("Ite\tMSE");
        RunningMessage[0] = "Ite\tMSE";

        for (int Ite = 0; Ite < 10000; Ite++) {
            double SquareErr = 0.0;
            BatchStore.clearNetwork();
            for (int i = 0; i < TrainData.size(); i++) {
                Double[] temp = (TrainData.get(i));
                double[][] InputVec = new double[InputNum][1];
                for (int k = 0; k < temp.length - 1; k++) {
                    InputVec[k][0] = temp[k];
                }
                double Tag = temp[temp.length - 1];

                ForwardResult = FinalANN.getForwardResult(InputVec);
                ErrVec[0][0] = Tag - ForwardResult[0][0];
                //SquareErr += ErrVec[0][0] * ErrVec[0][0];
                FinalANN.updateWeightNetwork(FinalANN.getSDBackwardUpdates(ErrVec, 0.01));
            }
            //BatchStore.averageNetwork(TrainData.size());
            //FinalANN.updateWeightNetwork(BatchStore.getANN());
            for (int i = 0; i < TrainData.size(); i++) {
                Double[] temp = (TrainData.get(i));
                double[][] InputVec = new double[InputNum][1];
                for (int k = 0; k < temp.length - 1; k++) {
                    InputVec[k][0] = temp[k];
                }
                double Tag = temp[temp.length - 1];
                ForwardResult = FinalANN.getForwardResult(InputVec);
                ErrVec[0][0] = Tag - ForwardResult[0][0];
                SquareErr += ErrVec[0][0] * ErrVec[0][0];
            }
            System.out.println(Ite + "\t" + (SquareErr / TrainData.size()));
            RunningMessage[Ite] = String.valueOf(Ite) + "\t" + String.valueOf(SquareErr / TrainData.size());
            if (SquareErr / TrainData.size() < MSE_upbound) {
                break;
            }
        }
        FileReadNWrite.LocalWriteFile("/home/mlx/Documents/ProcessResult/TEP_SDBP_IteMsg-v2", RunningMessage);

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
            //System.out.println(TestForwardResult[0][0]);
            TestResult[i] = String.valueOf(TestForwardResult[0][0]);
            TestErrSum += (Tag - TestForwardResult[0][0]) * (Tag - TestForwardResult[0][0]);
        }
        TestErrSum /= TestData.size();
        System.out.println("\n" + String.valueOf(TestErrSum));

        String[] SaveANN = FinalANN.saveANN();
        FileReadNWrite.LocalWriteFile("/home/mlx/Documents/ResultANN/TEP_SDBP_FinalANN-v2", SaveANN);
        FileReadNWrite.LocalWriteFile("/home/mlx/Documents/ProcessResult/TEP_SDBP_PreRe-v2", TestResult);

    }
}
