package MapReduce;

import HDFS_IO.ReadNWrite;
import NeuralNetwork.ArtificialNeuralNetwork;
import NeuralNetwork.LinearSearchMinumum;
import NeuralNetwork.NeuronLayer;
import org.apache.commons.math3.geometry.euclidean.threed.Line;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.yarn.util.SystemClock;
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec;
import org.jboss.netty.util.ExternalResourceReleasable;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.StringTokenizer;
import java.util.Vector;

/**
 * Created by mlx on 3/30/16.
 */
public class CGBPTrain_Map extends
        Mapper<LongWritable, Text, Text, DoubleWritable> {

    private Text Windex = new Text();
    private Text Bindex = new Text();
    private String ANN_path = "";
    private double ErrorUpperBound = 0.001;
    private double LeastGradientLength = 0.001;


    protected void getANNPath(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        this.ANN_path = conf.get("ThisIterationPath");
    }

    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {

        getANNPath(context);

        //pure text to String
        String line = value.toString();

        // 将输入的数据首先按行进行分割
        StringTokenizer tokenizerArticle = new StringTokenizer(line, "\n");

        //establish a ANN from existing file
        ArtificialNeuralNetwork TrainingANN = new ArtificialNeuralNetwork(ANN_path);
        ArtificialNeuralNetwork BatchStore = new ArtificialNeuralNetwork(TrainingANN.getANN());
        ArtificialNeuralNetwork GradientStore = new ArtificialNeuralNetwork(TrainingANN.getANN());
        BatchStore.clearNetwork();
        GradientStore.clearNetwork();

        int ParaNum = 0;
        for (int i = 0; i < TrainingANN.getLayerNum(); i++) {
            ParaNum += TrainingANN.getANN()[i].getNeuronNum() * (TrainingANN.getANN()[i].getInputNum() + 1);
        }

        double HeuristicStep = 0.1;

        NeuronLayer[] LastGenerationGradient = null;
        NeuronLayer[] LastGenerationDirection = null;

        Vector<Double[]> InputDataVector = new Vector<Double[]>();
        double SquErrSum = 0;
        int EntryNum = 0;
        //Data+Tag,transfer them into double
        for (; tokenizerArticle.hasMoreElements(); EntryNum++) {
            String[] DataArr = tokenizerArticle.nextToken().split("\t");
            Double[] WholeInput = new Double[DataArr.length];

            double Tag = Double.parseDouble(DataArr[DataArr.length - 1]);
            double[][] InputVec = new double[DataArr.length - 1][1];
            for (int i = 0; i < DataArr.length - 1; i++) {
                InputVec[i][0] = Double.parseDouble(DataArr[i]);
                WholeInput[i] = InputVec[i][0];
            }
            WholeInput[DataArr.length - 1] = Tag;
            InputDataVector.add(WholeInput);

            double[][] ForwardResult = TrainingANN.getForwardResult(InputVec);
            double[][] ErrVec = new double[TrainingANN.getOutputNum()][1];
            ErrVec[0][0] = Tag - ForwardResult[0][0];

            GradientStore.updateWeightNetwork(TrainingANN.getErrorGradient(ErrVec));

            SquErrSum += ErrVec[0][0] * ErrVec[0][0];
            context.write(new Text("SquareError"), new DoubleWritable(ErrVec[0][0] * ErrVec[0][0]));
        }
        double MSE = SquErrSum / EntryNum;
//            for (int i = 0; i < ThisGenerationGradient.length; i++) {
//                for (int j = 0; j < ThisGenerationGradient[i].getNeuronNum(); j++) {
//                    for (int k = 0; k < ThisGenerationGradient[i].getInputNum(); k++) {
//                        String WeightIndex = "W-" + String.valueOf(i) + "-";
//                        WeightIndex += String.valueOf(j) + "-" + String.valueOf(k);
//                        Windex.set(WeightIndex);
//                        context.write(Windex, new DoubleWritable(ThisGenerationGradient[i].getCertainWeight(j, k)));
//                    }
//                }
//            }
//            for (int i = 0; i < ThisGenerationGradient.length; i++) {
//                for (int j = 0; j < ThisGenerationGradient[i].getNeuronNum(); j++) {
//                    String BiasIndex = "B-" + String.valueOf(i) + "-";
//                    BiasIndex += String.valueOf(j);
//                    Bindex.set(BiasIndex);
//                    context.write(Bindex, new DoubleWritable(ThisGenerationGradient[i].getCertainBias(j)));
//                }
//            }
//            context.write(new Text("SquareError"), new DoubleWritable(ErrVec[0][0] * ErrVec[0][0]));
        NeuronLayer[] UpdatesDirection = null;
        NeuronLayer[] SearchDirection = null;
        for (int LineNum = 0; MSE >= ErrorUpperBound; LineNum++) {
            GradientStore.averageNetwork(EntryNum);
            NeuronLayer[] ThisGenerationGradient = GradientStore.getANN();
            if (ArtificialNeuralNetwork.getNeuronLayerArrLength_norm2(ThisGenerationGradient) < LeastGradientLength) {
                break;
            }
            if (LineNum % ParaNum == 0) {
                SearchDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(ThisGenerationGradient, -1.0);
            } else {
                SearchDirection = ArtificialNeuralNetwork.getFR_CGD(ThisGenerationGradient, LastGenerationGradient, LastGenerationDirection);
            }
            double[] IntervalLocation = LinearSearchMinumum.getIntervalLocation(TrainingANN, InputDataVector, SearchDirection, HeuristicStep);
            double OptimumLearningRate = LinearSearchMinumum.getTolerableMinimum(TrainingANN, InputDataVector, SearchDirection, IntervalLocation, 0.02);
            UpdatesDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, OptimumLearningRate);

            TrainingANN.updateWeightNetwork(UpdatesDirection);
            BatchStore.updateWeightNetwork(UpdatesDirection);
//            ForwardResult = TrainingANN.getForwardResult(InputVec);
//            ErrVec[0][0] = Tag - ForwardResult[0][0];
            SquErrSum = 0;
            GradientStore.clearNetwork();
            for (int i = 0; i < InputDataVector.size(); i++) {
                double[][] InputVec = new double[TrainingANN.getInputNum()][1];
                for (int j = 0; j < TrainingANN.getInputNum(); j++) {
                    InputVec[j][0] = ((Double[]) (InputDataVector.get(i)))[j];
                }
                double[][] ForwardResult1 = TrainingANN.getForwardResult(InputVec);

                double[][] ErrVec = new double[TrainingANN.getOutputNum()][1];
                ErrVec[0][0] = ((Double[]) (InputDataVector.get(i)))[TrainingANN.getInputNum()] - ForwardResult1[0][0];
                GradientStore.updateWeightNetwork(TrainingANN.getErrorGradient(ErrVec));

                SquErrSum += Math.pow(ErrVec[0][0], 2);
            }
            MSE = SquErrSum / EntryNum;

            LastGenerationGradient = ThisGenerationGradient;
            LastGenerationDirection = SearchDirection;
        }

        for (int i = 0; i < BatchStore.getANN().length; i++) {
            for (int j = 0; j < BatchStore.getANN()[i].getNeuronNum(); j++) {
                for (int k = 0; k < BatchStore.getANN()[i].getInputNum(); k++) {
                    String WeightIndex = "W-" + String.valueOf(i) + "-";
                    WeightIndex += String.valueOf(j) + "-" + String.valueOf(k);
                    Windex.set(WeightIndex);
                    context.write(Windex, new DoubleWritable(BatchStore.getANN()[i].getCertainWeight(j, k)));
                }
            }
        }
        for (int i = 0; i < BatchStore.getANN().length; i++) {
            for (int j = 0; j < BatchStore.getANN()[i].getNeuronNum(); j++) {
                String BiasIndex = "B-" + String.valueOf(i) + "-";
                BiasIndex += String.valueOf(j);
                Bindex.set(BiasIndex);
                context.write(Bindex, new DoubleWritable(BatchStore.getANN()[i].getCertainBias(j)));
            }
        }
        //BatchStore.clearNetwork();
    }
}

