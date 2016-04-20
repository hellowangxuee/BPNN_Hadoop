package MapReduce;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Created by mlx on 4/10/16.
 */
public class ValidationReduce extends
        Reducer<Text,DoubleWritable,Text,DoubleWritable> {
    public void reduce(Text key, Iterable<DoubleWritable> values, Context context)
            throws IOException, InterruptedException {
        int num = 0;
        double SumOfSquareError = 0.0;
        for (DoubleWritable val : values) {
            num++;
            SumOfSquareError += val.get();
        }
        context.write(new Text("MeanSquareError"), new DoubleWritable(SumOfSquareError / num));
    }
}
