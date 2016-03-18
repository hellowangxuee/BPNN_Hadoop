package MapReduce;

import Jampack.*;
import org.apache.commons.io.IOExceptionWithCause;
import org.apache.commons.math3.geometry.Vector;
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

    protected void getANNPath(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        ANN_path = conf.get("ThisIterationPath");
    }

    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {

        getANNPath(context);

        //pure text to String
        String line = value.toString();

        // 将输入的数据首先按行进行分割
        StringTokenizer tokenizerArticle = new StringTokenizer(line, "\n");

        //Data+Tag,transfer them into double
        while (tokenizerArticle.hasMoreElements()) {
            String[] DataArr = tokenizerArticle.nextToken().split("\t");
            double Tag = Double.parseDouble(DataArr[DataArr.length - 1]);
            double[][] InputVec = new double[DataArr.length - 1][1];
            for (int i = 0; i < DataArr.length - 1; i++) {
                InputVec[i][0] = Double.parseDouble(DataArr[i]);
            }


            //learning rate
            double alpha = 0.1;

            //establish a ANN from existing file
            ArtificialNeuralNetwork TrainingANN = new ArtificialNeuralNetwork(ANN_path);
//            int InputNum = 1;
//            int LayerNum = 2;
//            int[] NumEachLayer = {3, 1};
//            int[] IndexEachLayer = {1, 3};
//
//            ArtificialNeuralNetwork TrainingANN = new ArtificialNeuralNetwork(InputNum, LayerNum, NumEachLayer, IndexEachLayer);

            double[][] ForwardResult = TrainingANN.getForwardResult(InputVec);
            double[][] ErrVec = new double[ForwardResult.length][1];

            ErrVec[0][0] = Tag - ForwardResult[0][0];

            NeuronLayer[] WeightChangeArr = TrainingANN.getBackwardChange(ErrVec, alpha);

            for (int i = 0; i < WeightChangeArr.length; i++) {
                for (int j = 0; j < WeightChangeArr[i].getNeuronNum(); j++) {
                    for (int k = 0; k < WeightChangeArr[i].getInputNum(); k++) {
                        String WeightIndex = "W-" + String.valueOf(i) + "-";
                        WeightIndex += String.valueOf(j) + "-" + String.valueOf(k);
                        Windex.set(WeightIndex);
                        context.write(Windex, new DoubleWritable(WeightChangeArr[i].getCertainWeight(j, k)));
                    }
                }
            }

            for (int i = 0; i < WeightChangeArr.length; i++) {
                for (int j = 0; j < WeightChangeArr[i].getNeuronNum(); j++) {
                    String BiasIndex = "B-" + String.valueOf(i) + "-";
                    BiasIndex += String.valueOf(j);
                    Bindex.set(BiasIndex);
                    context.write(Bindex, new DoubleWritable(WeightChangeArr[i].getCertainBias(j)));
                }
            }
            context.write(new Text("SquareError"), new DoubleWritable(ErrVec[0][0] * ErrVec[0][0]));
        }
    }
}
