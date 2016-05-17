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
    private double LeastGradientLength = 1E-6;


    protected void getANNPath(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        this.ANN_path = conf.get("ThisIterationPath");
    }

    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        getANNPath(context);
        String line = value.toString();
        String[] LineDataArr = line.split(";");
        Vector<Double[]> InputPair = new Vector<Double[]>();
        for (int k = 0; k < LineDataArr.length; k++) {
            String[] OneEntry = LineDataArr[k].split("\t");
            Double[] EntryArr = new Double[OneEntry.length];
            for (int t = 0; t < OneEntry.length; t++) {
                EntryArr[t] = Double.parseDouble(OneEntry[t]);
            }
            InputPair.add(EntryArr);
        }

        //establish a ANN from existing file
        ArtificialNeuralNetwork TrainingANN = new ArtificialNeuralNetwork(ANN_path);
        ArtificialNeuralNetwork BatchStore = new ArtificialNeuralNetwork(TrainingANN.getANN());
        ArtificialNeuralNetwork TotalUpdates = new ArtificialNeuralNetwork(TrainingANN.getANN());
        TotalUpdates.clearNetwork();
        int ParaNum = 0;
        for (int i = 0; i < TrainingANN.getLayerNum(); i++) {
            ParaNum += TrainingANN.getANN()[i].getNeuronNum() * (TrainingANN.getANN()[i].getInputNum() + 1);
        }

        double HeuristicStep = 0.1;
        NeuronLayer[] LastGenerationGradient = null;
        NeuronLayer[] LastGenerationDirection = null;
        double[][] ForwardResult = null;
        double[][] ErrVec = new double[TrainingANN.getOutputNum()][1];

        for (int time = 0; time < ParaNum; time++) {
            double SE = 0;
            BatchStore.clearNetwork();
            for (int p = 0; p < InputPair.size(); p++) {
                Double[] OneEntry = InputPair.get(p);
                double Tag = OneEntry[OneEntry.length - 1];
                double[][] InputVec = new double[TrainingANN.getInputNum()][1];
                for (int k = 0; k < OneEntry.length - 1; k++) {
                    InputVec[k][0] = OneEntry[k];
                }
                ForwardResult = TrainingANN.getForwardResult(InputVec);
                ErrVec[0][0] = Tag - ForwardResult[0][0];
                SE += Math.pow(ErrVec[0][0], 2);
                BatchStore.updateWeightNetwork(TrainingANN.getErrorGradient(ErrVec));
            }
            if (time == 0) {
                context.write(new Text("SquareError"), new DoubleWritable(SE / InputPair.size()));
            }

            BatchStore.averageNetwork(InputPair.size());
            NeuronLayer[] ThisGenerationGradient = BatchStore.getANN();
            NeuronLayer[] UpdatesDirection = null;
            NeuronLayer[] SearchDirection = null;
            double LengthOfGradient = ArtificialNeuralNetwork.getNeuronLayerArrLength_norm2(ThisGenerationGradient);
            if ((SE / InputPair.size() < ErrorUpperBound) || (LengthOfGradient < LeastGradientLength)) {
                break;
            }

            if (time % ParaNum == 0) {
                SearchDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(ThisGenerationGradient, -1.0);
            } else {
                SearchDirection = ArtificialNeuralNetwork.getFR_CGD(ThisGenerationGradient, LastGenerationGradient, LastGenerationDirection);
            }
            double[] IntervalLocation = LinearSearchMinumum.getIntervalLocation(TrainingANN, InputPair, SearchDirection, HeuristicStep);
            double OptimumLearningRate = LinearSearchMinumum.getTolerableMinimum(TrainingANN, InputPair, SearchDirection, IntervalLocation, 0.02);
            UpdatesDirection = ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, OptimumLearningRate);
            TrainingANN.updateWeightNetwork(UpdatesDirection);
            TotalUpdates.updateWeightNetwork(UpdatesDirection);
            LastGenerationGradient = ThisGenerationGradient;
            LastGenerationDirection = SearchDirection;
        }

        for (int i = 0; i < TotalUpdates.getANN().length; i++) {
            for (int j = 0; j < TotalUpdates.getANN()[i].getNeuronNum(); j++) {
                for (int k = 0; k < TotalUpdates.getANN()[i].getInputNum(); k++) {
                    String WeightIndex = "W-" + String.valueOf(i) + "-";
                    WeightIndex += String.valueOf(j) + "-" + String.valueOf(k);
                    Windex.set(WeightIndex);
                    context.write(Windex, new DoubleWritable(TotalUpdates.getANN()[i].getCertainWeight(j, k)));

                }
            }
        }
        for (int i = 0; i < TotalUpdates.getANN().length; i++) {
            for (int j = 0; j < TotalUpdates.getANN()[i].getNeuronNum(); j++) {
                String BiasIndex = "B-" + String.valueOf(i) + "-";
                BiasIndex += String.valueOf(j);
                Bindex.set(BiasIndex);
                context.write(Bindex, new DoubleWritable(TotalUpdates.getANN()[i].getCertainBias(j)));
            }
        }
    }
}

