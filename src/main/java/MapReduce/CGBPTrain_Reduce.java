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
 * Created by mlx on 3/30/16.
 */
public class CGBPTrain_Reduce extends
        Reducer<Text,DoubleWritable,Text,DoubleWritable> {

    public void reduce(Text key, Iterable<DoubleWritable> values, Context context)
            throws IOException, InterruptedException {
        String keyStr = key.toString();
        if (keyStr.equals("SquareError")) {
            int num = 0;
            double SumOfSquareError = 0.0;
            for (DoubleWritable val : values) {
                num++;
                SumOfSquareError += val.get();
            }
            context.write(new Text("MeanSquareError"), new DoubleWritable(SumOfSquareError / num));
        } else {

            int WeightCount = 0;
            double SumOfWeightChanges = 0.0;
            for (DoubleWritable val : values) {
                WeightCount++;
                SumOfWeightChanges += val.get();
            }
            context.write(key, new DoubleWritable(SumOfWeightChanges / WeightCount));
        }
    }
}