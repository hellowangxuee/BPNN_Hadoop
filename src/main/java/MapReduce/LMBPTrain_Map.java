package MapReduce;

import NeuralNetwork.ArtificialNeuralNetwork;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.StringTokenizer;

/**
 * Created by mlx on 4/4/16.
 */
public class LMBPTrain_Map extends
        Mapper<LongWritable, Text, Text, DoubleWritable> {

    private String ANN_path = "";
    private double miu = 0.01;
    private double MultipliedFactor = 10;

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
        double ThisTimeSquErrSum=0.0;
        double LastTimeSquErrSum=0.0;
        while(ThisTimeSquErrSum<LastTimeSquErrSum) {
            for (; tokenizerArticle.hasMoreElements(); ) {



            }
        }
    }
}
