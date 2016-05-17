package MapReduce;

import Jampack.Rand;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Random;

/**
 * Created by mlx on 4/25/16.
 */
public class BayRegBPTrain_Reduce extends
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
            if(Math.abs(SumOfWeightChanges / WeightCount ) <= 10){
                context.write(key, new DoubleWritable(SumOfWeightChanges / WeightCount));
            }
            else {
                Random r=new Random();
                context.write(key, new DoubleWritable(Math.sqrt(0.01)* r.nextGaussian()));
            }
        }
    }
}
