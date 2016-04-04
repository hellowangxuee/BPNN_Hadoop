package MapReduce;

import org.apache.commons.math3.stat.descriptive.summary.Sum;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Map;

/**
 * Created by mlx on 3/11/16.
 */
public class BPTrain_Reduce extends
        Reducer<Text,DoubleWritable,Text,DoubleWritable> {

    private String lastUpdatePath = null;
    private double Momentum = 0.0;
    private Map<String, Double> LastUpdateMap = null;

    protected void getLastUpdatePath(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        this.lastUpdatePath = conf.get("LastUpdatePath");
        this.Momentum = conf.getDouble("Momentum", 0.0);
        if (this.lastUpdatePath != null && this.Momentum != 0.0) {
            this.LastUpdateMap = BPTrain.getWeightUpdatesFromFile(this.lastUpdatePath);
        }
    }

    public void reduce(Text key, Iterable<DoubleWritable> values, Context context)
            throws IOException, InterruptedException {
        getLastUpdatePath(context);
        if (this.lastUpdatePath == null || this.Momentum == 0.0) {
            if (key.toString().equals("SquareError")) {
                int num = 0;
                double SumOfSquareError = 0.0;
                for (DoubleWritable val : values) {
                    num++;
                    SumOfSquareError += val.get();
                }
                context.write(new Text("MeanSquareError"), new DoubleWritable(SumOfSquareError / num));
            } else {
                int NumOfWeightChanges = 0;
                double SumOfWeightChanges = 0.0;

                for (DoubleWritable val : values) {
                    NumOfWeightChanges++;
                    SumOfWeightChanges += val.get();
                }
                double AveOfWeightChanges = SumOfWeightChanges / NumOfWeightChanges;

                context.write(key, new DoubleWritable(AveOfWeightChanges));
            }
        } else {
            if (key.toString().equals("SquareError")) {
                int num = 0;
                double SumOfSquareError = 0.0;
                for (DoubleWritable val : values) {
                    num++;
                    SumOfSquareError += val.get();
                }
                context.write(new Text("MeanSquareError"), new DoubleWritable(SumOfSquareError / num));
            } else {
                int NumOfWeightChanges = 0;
                double SumOfWeightChanges = 0.0;

                for (DoubleWritable val : values) {
                    NumOfWeightChanges++;
                    SumOfWeightChanges += val.get();
                }
                double AveOfWeightChanges = SumOfWeightChanges / NumOfWeightChanges;
                double LastUpdateValue = this.LastUpdateMap.get(key.toString());


                context.write(key, new DoubleWritable(this.Momentum * LastUpdateValue + (1 - this.Momentum) * AveOfWeightChanges));
            }
        }
    }
}