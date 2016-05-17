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
 * Created by mlx on 4/10/16.
 */
public class ValidationMap extends
        Mapper<LongWritable, Text, Text, DoubleWritable> {

    private String ANN_path = "";
    protected void getANNPath(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        this.ANN_path = conf.get("ThisIterationPath");
    }
    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        getANNPath(context);

        //pure text to String
        String line = value.toString();
        String[] LineDataArr = line.split(";");
        //establish a ANN from existing file
        ArtificialNeuralNetwork TrainingANN = new ArtificialNeuralNetwork(ANN_path);

        // 将输入的数据首先按行进行分割
        double SE = 0.0;
        double[][] ForwardResult = null;
        for (int EntryNum = 0; EntryNum < LineDataArr.length; EntryNum++) {
            String[] DataArr = LineDataArr[EntryNum].split("\t");
            double Tag = Double.parseDouble(DataArr[DataArr.length - 1]);
            double[][] InputVec = new double[DataArr.length - 1][1];
            for (int i = 0; i < DataArr.length - 1; i++) {
                InputVec[i][0] = Double.parseDouble(DataArr[i]);
            }
            ForwardResult = TrainingANN.getForwardResult(InputVec);
            SE += (Tag - ForwardResult[0][0]) * (Tag - ForwardResult[0][0]);
        }
        context.write(new Text("SquareError"), new DoubleWritable(SE / LineDataArr.length));
    }
}
