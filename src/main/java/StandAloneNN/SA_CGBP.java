package StandAloneNN;

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
        Vector<Double[]> InputPair = readTxtFile("/home/mlx/Documents/D1");
        Vector<Double[]> TestPair = readTxtFile("/home/mlx/Documents/TestingDataset");

        double ErrorUpperBound = 0.05;
        double LeastGradientLength = 1E-5;
        double HeuristicStep = 0.1;

        int InputNum = 1;
        int LayerNum = 3;
        int[] NumEachLayer = {7, 10, 1};
        int[] IndexEachLayer = {1, 1, 3};
        //ArtificialNeuralNetwork FinalANN = new ArtificialNeuralNetwork("hdfs://Master:9000/user/mlx/StartANN");
        ArtificialNeuralNetwork FinalANN = new ArtificialNeuralNetwork(InputNum, LayerNum, NumEachLayer, IndexEachLayer);

        //ArtificialNeuralNetwork InitialANN = new ArtificialNeuralNetwork(FinalANN.getANN());
        ArtificialNeuralNetwork BatchStore = new ArtificialNeuralNetwork(FinalANN.getANN());
        ArtificialNeuralNetwork GradientStore = new ArtificialNeuralNetwork(FinalANN.getANN());

        int ParaNum = 0;
        for (int i = 0; i < FinalANN.getLayerNum(); i++) {
            ParaNum += FinalANN.getANN()[i].getNeuronNum() * (FinalANN.getANN()[i].getInputNum() + 1);
        }

        double[][] ForwardResult=null;
        double[][] ErrVec = new double[FinalANN.getOutputNum()][1];
        NeuronLayer[] LastGenerationGradient = null;
        NeuronLayer[] LastGenerationDirection = null;

        HashSet h = new HashSet();

        int PackageSize = 100;

        Vector<Vector<Double[]>> InputPairPackage = new Vector<Vector<Double[]>>();

        for (int i = 0; i < InputPair.size(); ) {
            Vector<Double[]> OnePackInput = new Vector<Double[]>();
            for (int j = 0; j < PackageSize && i < InputPair.size(); j++, i++) {
                int Random=getRandomNum(0,InputPair.size()-1);
                while (!h.contains(Random)){
                    Random=getRandomNum(0,InputPair.size()-1);
                    h.add(Random);
                }
                OnePackInput.add(InputPair.get(Random));
            }
            InputPairPackage.add(OnePackInput);
        }

        for(int time=0;time<10;time++) {
            double TotalMSE=0;
            BatchStore.clearNetwork();
            for (int p = 0; p < InputPairPackage.size(); p++) {
                ArtificialNeuralNetwork InitialANN=new ArtificialNeuralNetwork(FinalANN.getANN());
                Vector<Double[]> OnePackInput = InputPairPackage.get(p);
                double MSE = 0;
                for (int Ite = 0; Ite == 0 || MSE >= ErrorUpperBound; Ite++) {
                    MSE = 0;
                    GradientStore.clearNetwork();
                    double Tag = 0;
                    for (int i = 0; i < OnePackInput.size(); i++) {
                        Double[] OneEntry = OnePackInput.get(i);
                        Tag = OneEntry[OneEntry.length - 1];
                        double[][] InputVec = new double[InputNum][1];
                        for (int k = 0; k < OneEntry.length - 1; k++) {
                            InputVec[k][0] = OneEntry[k];
                        }
                        ForwardResult = InitialANN.getForwardResult(InputVec);
                        ErrVec[0][0] = Tag - ForwardResult[0][0];
                        GradientStore.updateWeightNetwork(InitialANN.getErrorGradient(ErrVec));
                        MSE += Math.pow(ErrVec[0][0], 2);
                    }

                    MSE /= OnePackInput.size();
                    NeuronLayer[] UpdatesDirection = null;
                    NeuronLayer[] SearchDirection = null;
                    GradientStore.averageNetwork(OnePackInput.size());
                    NeuronLayer[] ThisGenerationGradient = GradientStore.getANN();
                    if (ArtificialNeuralNetwork.getNeuronLayerArrLength_norm2(ThisGenerationGradient) < LeastGradientLength) {
                        break;
                    }
                    if (Ite % ParaNum == 0) {
                        SearchDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(ThisGenerationGradient, -1.0);
                    } else {
                        SearchDirection = ArtificialNeuralNetwork.getFR_CGD(ThisGenerationGradient, LastGenerationGradient, LastGenerationDirection);
                    }
                    double[] IntervalLocation = LinearSearchMinumum.getIntervalLocation(InitialANN, OnePackInput, SearchDirection, HeuristicStep);
                    double OptimumLearningRate = LinearSearchMinumum.getTolerableMinimum(InitialANN, OnePackInput, SearchDirection, IntervalLocation, 0.02);
                    UpdatesDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, OptimumLearningRate);
                    InitialANN.updateWeightNetwork(UpdatesDirection);
                    BatchStore.updateWeightNetwork(UpdatesDirection);
                    LastGenerationGradient = ThisGenerationGradient;
                    LastGenerationDirection = SearchDirection;
                    //System.out.println(String.valueOf(p) + "\t" + String.valueOf(Ite) +  "\t" + String.valueOf(MSE));
                }
                TotalMSE += MSE;
                //System.out.println(String.valueOf(p)  + "\t" + String.valueOf(TotalMSE));
            }
            BatchStore.averageNetwork(InputPairPackage.size());
            FinalANN.updateWeightNetwork(BatchStore.getANN());
            TotalMSE/=InputPairPackage.size();
            System.out.println(String.valueOf(time)+"\t"+ String.valueOf(TotalMSE));
        }

//        double TestMSE=0;
//        int RightNum=0;
//        int WrongNum=0;
//        for(int i=0;i<TestPair.size();i++) {
//            Double[] OneEntry = TestPair.get(i);
//            double Tag = OneEntry[OneEntry.length - 1]+1;
//            double[][] InputVec = new double[InputNum][1];
//            for (int k = 0; k < OneEntry.length - 1; k++) {
//                InputVec[k][0] = OneEntry[k];
//            }
//            ForwardResult = FinalANN.getForwardResult(InputVec);
//            if(ForwardResult[0][0] > 0.5){
//                if(Tag==1.0){
//                    RightNum++;
//                }
//                else{
//                    WrongNum++;
//                }
//            }
//            else{
//                if(Tag==1.0){
//                    WrongNum++;
//                }
//                else{
//                    RightNum++;
//                }
//            }
//            ErrVec[0][0] = Tag - ForwardResult[0][0];
//            TestMSE += ErrVec[0][0] * ErrVec[0][0];
//            System.out.println(String.valueOf(Tag)+"\t"+String.valueOf(ForwardResult[0][0]));
//        }
//        TestMSE/=TestPair.size();
//        System.out.println(String.valueOf(TestMSE)+"\t"+String.valueOf(RightNum)+"\t"+String.valueOf(WrongNum));
//        for (int Ite = 0; Ite < 20; Ite++) {
//            BatchStore.clearNetwork();
//            double TotalSE = 0;
//            int RandomNum = getRandomNum(0, InputPair.size() - 1);
//            h.add(RandomNum);
//            for (int i = 0; i < InputPair.size(); i++) {
//                while (!h.contains(RandomNum)) {
//                    RandomNum = getRandomNum(0, InputPair.size() - 1);
//                    h.add(RandomNum);
//                }
//                Double[] OneEntry = InputPair.get(RandomNum);
//                GradientStore.clearNetwork();
//                double Tag = OneEntry[OneEntry.length - 1] ;
//
//                double[][] InputVec = new double[InputNum][1];
//                for (int k = 0; k < OneEntry.length - 1; k++) {
//                    InputVec[k][0] = OneEntry[k];
//                }
//                ForwardResult = FinalANN.getForwardResult(InputVec);
//                double[][] ErrVec = new double[FinalANN.getOutputNum()][1];
//                ErrVec[0][0] = Tag - ForwardResult[0][0];
//
//                GradientStore.updateWeightNetwork(FinalANN.getErrorGradient(ErrVec));
//                TotalSE += ErrVec[0][0] * ErrVec[0][0];
//
//                NeuronLayer[] LastGenerationGradient = null;
//                NeuronLayer[] LastGenerationDirection = null;
//                NeuronLayer[] UpdatesDirection = null;
//                NeuronLayer[] SearchDirection = null;
//
//                Vector<Double[]> oneVec = new Vector<Double[]>();
//                oneVec.add(OneEntry);
//                ArtificialNeuralNetwork InitialANN = new ArtificialNeuralNetwork(FinalANN.getANN());
//                for (int Ite = 0; Math.abs(ErrVec[0][0]) >= ErrorUpperBound  ; Ite++) {
//                    //GradientStore.averageNetwork(InputPair.size());
//                    NeuronLayer[] ThisGenerationGradient = GradientStore.getANN();
//                    if (ArtificialNeuralNetwork.getNeuronLayerArrLength_norm2(ThisGenerationGradient) < LeastGradientLength) {
//                        break;
//                    }
//                    if (Ite % ParaNum == 0) {
//                        SearchDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(ThisGenerationGradient, -1.0);
//                    } else {
//                        SearchDirection = ArtificialNeuralNetwork.getFR_CGD(ThisGenerationGradient, LastGenerationGradient, LastGenerationDirection);
//                    }
//                    double[] IntervalLocation = LinearSearchMinumum.getIntervalLocation(InitialANN, oneVec, SearchDirection, HeuristicStep);
//                    double OptimumLearningRate = LinearSearchMinumum.getTolerableMinimum(InitialANN, oneVec, SearchDirection, IntervalLocation, 0.02);
//                    UpdatesDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, OptimumLearningRate);
//
//                    InitialANN.updateWeightNetwork(UpdatesDirection);
//                    BatchStore.updateWeightNetwork(UpdatesDirection);
//                    ForwardResult = InitialANN.getForwardResult(InputVec);
//                    ErrVec[0][0] = Tag - ForwardResult[0][0];
//                    if(Math.abs(ErrVec[0][0])>2){
//                        break;
//                    }
//                    LastGenerationGradient = ThisGenerationGradient;
//                    LastGenerationDirection = SearchDirection;
//                }
//                RandomNum = getRandomNum(0, InputPair.size() - 1);
//                h.add(RandomNum);
//                //InitialANN.updateWeightNetwork()
//                //FinalANN.updateWeightNetwork(BatchStore.getANN());
//            }
//            BatchStore.averageNetwork(InputPair.size());
//            FinalANN.updateWeightNetwork(BatchStore.getANN());
//            System.out.println(String.valueOf(Ite)+"\t"+String.valueOf(TotalSE));
//            h.clear();
////            String[] FANN = FinalANN.saveANN();
////            for (int i = 0; i < FANN.length; i++) {
////                System.out.println(FANN[i]);
////            }
//        }
    }
    public static int getRandomNum(int m,int n) {
        return (m + (int) (Math.random() * (n - m + 1)));
    }
}
