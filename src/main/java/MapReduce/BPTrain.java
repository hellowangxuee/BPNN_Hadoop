package MapReduce;

import NeuralNetwork.ArtificialNeuralNetwork;
import NeuralNetwork.NeuronLayer;
import org.apache.hadoop.conf.Configuration;
import MapReduce.BPTrain_Map;
import MapReduce.BPTrain_Reduce;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;
import java.util.StringTokenizer;

/**
 * Created by mlx on 3/14/16.
 */
public class BPTrain {

    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();
        Job job = new Job(conf);
        job.setJarByClass(BPTrain.class);
        job.setJobName("BPTrain");

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        job.setMapperClass(BPTrain_Map.class);
        job.setReducerClass(BPTrain_Reduce.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.waitForCompletion(true);
    }

}
