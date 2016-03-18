package MapReduce;

import org.apache.commons.math3.stat.descriptive.summary.Sum;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Created by mlx on 3/11/16.
 */
public class BPTrain_Reduce extends
        Reducer<Text,DoubleWritable,Text,DoubleWritable> {

    public void reduce(Text key, Iterable<DoubleWritable> values, Context context)
            throws IOException, InterruptedException {
        if (key.toString().equals("SquareError")) {
            double SumOfSquareError = 0.0;
            for (DoubleWritable val : values) {
                SumOfSquareError += val.get();
            }
            context.write(new Text("TotalSquareError"), new DoubleWritable(SumOfSquareError));
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
    }
}