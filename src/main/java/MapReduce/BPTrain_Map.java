package MapReduce;

import Jampack.*;
import org.apache.commons.io.IOExceptionWithCause;
import org.apache.commons.math3.geometry.Vector;
import org.apache.commons.math3.geometry.euclidean.threed.Line;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.JobConfigurable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.join.StreamBackedIterator;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.net.URLDecoder;
import java.util.StringTokenizer;

import NeuralNetwork.*;
import FileIO.*;
import org.apache.jasper.JasperException;

/**
 * Created by Jackie on 16/3/3.
 */
public class BPTrain_Map extends
        Mapper<LongWritable, Text, Text, DoubleWritable> {

    private Text Windex = new Text();
    private Text Bindex = new Text();
    private String ANN_path = "";
    private double learning_rate = 0.1;

    protected void getANNPath(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        this.ANN_path = conf.get("ThisIterationPath");
        this.learning_rate = conf.getDouble("LearningRate", 0.1);
    }

    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        getANNPath(context);
        //pure text to String
        String line = value.toString();
        String[] LineDataArr = line.split(";");

        //establish a ANN from existing file
        ArtificialNeuralNetwork TrainingANN = new ArtificialNeuralNetwork(ANN_path);
        ArtificialNeuralNetwork TotalSDUpdates = new ArtificialNeuralNetwork(TrainingANN.getANN());
        TotalSDUpdates.clearNetwork();
        int ParaNum = 0;
        for (int i = 0; i < TrainingANN.getLayerNum(); i++) {
            ParaNum += TrainingANN.getANN()[i].getNeuronNum() * (TrainingANN.getANN()[i].getInputNum() + 1);
        }

        double[][] ErrVec = new double[TrainingANN.getOutputNum()][1];
        double[][] ForwardResult = null;

        double SquareError = 0.0;
        for (int EntryNum = 0; EntryNum < LineDataArr.length; EntryNum++) {
            String[] DataArr = LineDataArr[EntryNum].split("\t");
            double Tag = Double.parseDouble(DataArr[DataArr.length - 1]);
            double[][] InputVec = new double[DataArr.length - 1][1];
            for (int i = 0; i < DataArr.length - 1; i++) {
                InputVec[i][0] = Double.parseDouble(DataArr[i]);
            }
            ForwardResult = TrainingANN.getForwardResult(InputVec);
            SquareError += (Tag - ForwardResult[0][0]) * (Tag - ForwardResult[0][0]);
        }
        context.write(new Text("SquareError"), new DoubleWritable(SquareError / LineDataArr.length));

        for (int Ite = 0; Ite < ParaNum; Ite++) {
            for (int EntryNum = 0; EntryNum < LineDataArr.length; EntryNum++) {
                String[] DataArr = LineDataArr[EntryNum].split("\t");
                double Tag = Double.parseDouble(DataArr[DataArr.length - 1]);
                double[][] InputVec = new double[DataArr.length - 1][1];
                for (int i = 0; i < DataArr.length - 1; i++) {
                    InputVec[i][0] = Double.parseDouble(DataArr[i]);
                }
                ForwardResult = TrainingANN.getForwardResult(InputVec);
                ErrVec[0][0] = Tag - ForwardResult[0][0];
                NeuronLayer[] WeightChangeArr = TrainingANN.getSDBackwardUpdates(ErrVec, learning_rate);
                TrainingANN.updateWeightNetwork(WeightChangeArr);
                TotalSDUpdates.updateWeightNetwork(WeightChangeArr);
            }
        }
        for (int i = 0; i < TotalSDUpdates.getANN().length; i++) {
            for (int j = 0; j < TotalSDUpdates.getANN()[i].getNeuronNum(); j++) {
                for (int k = 0; k < TotalSDUpdates.getANN()[i].getInputNum(); k++) {
                    String WeightIndex = "W-" + String.valueOf(i) + "-";
                    WeightIndex += String.valueOf(j) + "-" + String.valueOf(k);
                    Windex.set(WeightIndex);
                    context.write(Windex, new DoubleWritable(TotalSDUpdates.getANN()[i].getCertainWeight(j, k)));
                }
            }
        }

        for (int i = 0; i < TotalSDUpdates.getANN().length; i++) {
            for (int j = 0; j < TotalSDUpdates.getANN()[i].getNeuronNum(); j++) {
                String BiasIndex = "B-" + String.valueOf(i) + "-";
                BiasIndex += String.valueOf(j);
                Bindex.set(BiasIndex);
                context.write(Bindex, new DoubleWritable(TotalSDUpdates.getANN()[i].getCertainBias(j)));
            }
        }
        //context.write(new Text("SquareError"), new DoubleWritable(ErrVec[0][0] * ErrVec[0][0]));
    }
}

