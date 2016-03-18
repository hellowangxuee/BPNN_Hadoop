package MapReduce;

import HDFS_IO.ReadNWrite;
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
import org.apache.hadoop.mapreduce.jobhistory.JobHistoryParser;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.v2.app.webapp.dao.JobInfo;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.net.URLEncoder;
import java.util.StringTokenizer;

/**
 * Created by mlx on 3/14/16.
 */
public class BPTrain {

    private static Job getSettedJob(Configuration conf,int IterationNum) throws Exception{
        Job job = new Job(conf);
        job.setJarByClass(BPTrain.class);
        job.setJobName("BPTrain" + String.valueOf(IterationNum));

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        job.setMapperClass(BPTrain_Map.class);
        job.setReducerClass(BPTrain_Reduce.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        return job;
    }

    public static void updateANN(String oldANNPath,String WeightChangePath,String SavePath) throws IOException{
        ArtificialNeuralNetwork oldANN=new ArtificialNeuralNetwork(oldANNPath);
        String[] WeightChangeArr= ReadNWrite.hdfs_Read(WeightChangePath);

        for(int i=0;i<WeightChangeArr.length;i++){
            String[] IndexNChange=WeightChangeArr[i].split("\t");
            double ChangeValue=Double.parseDouble(IndexNChange[1]);

            String[] IndexArr=IndexNChange[0].split("-");
            if(IndexArr.length==4) {
                int LayNum = Integer.parseInt(IndexArr[1]);
                int NeuronNum=Integer.parseInt(IndexArr[2]);
                int WeightNum=Integer.parseInt(IndexArr[3]);
                oldANN.updateCertainWeight(LayNum,NeuronNum,WeightNum,ChangeValue);
            }
            else if(IndexArr.length==3){
                int LayNum = Integer.parseInt(IndexArr[1]);
                int NeuronNum=Integer.parseInt(IndexArr[2]);
                oldANN.updateCertainBias(LayNum,NeuronNum,ChangeValue);
            }
            else{
                continue;
            }
        }

        String[] new_ANN_content=oldANN.saveANN();
        ReadNWrite.hdfs_Write(new_ANN_content,SavePath);

    }

    public static void main(String[] args) throws Exception {
        int IterationNum=50;
        int TotalIterationNum=60;
        String ipPrefix="hdfs://localhost:9000";

        String[] pathArr=args[0].split("/");
        String pathPrefix="";
        for(int i=1;i<pathArr.length-1;i++) {
            pathPrefix += "/" + pathArr[i];
        }

        for(;IterationNum<TotalIterationNum;IterationNum++) {
            Configuration conf = new Configuration();
            if (IterationNum==0) {
                conf.set("ThisIterationPath", ipPrefix+pathPrefix+"/testANN0");
            }
            else{
                String oldANNPath=ipPrefix+pathPrefix+"/testANN"+String.valueOf(IterationNum-1);
                String newANNPath=ipPrefix+pathPrefix+"/testANN"+String.valueOf(IterationNum);
                String ChangeValuePath=ipPrefix+args[1]+"-"+String.valueOf(IterationNum-1)+"/part-r-00000";

                BPTrain.updateANN(oldANNPath,ChangeValuePath,newANNPath);
                conf.set("ThisIterationPath",newANNPath);
            }

            //Set job configs
            Job job=getSettedJob(conf,IterationNum);
//            Job job = new Job(conf);
//            job.setJarByClass(BPTrain.class);
//            job.setJobName("BPTrain" + String.valueOf(IterationNum));
//
//            job.setOutputKeyClass(Text.class);
//            job.setOutputValueClass(DoubleWritable.class);
//
//            job.setMapperClass(BPTrain_Map.class);
//            job.setReducerClass(BPTrain_Reduce.class);
//
//            job.setInputFormatClass(TextInputFormat.class);
//            job.setOutputFormatClass(TextOutputFormat.class);

            FileInputFormat.addInputPath(job, new Path(args[0]));

            String outPath = args[1] + "-" + String.valueOf(IterationNum);
            FileOutputFormat.setOutputPath(job, new Path(outPath));

            job.waitForCompletion(true);
        }
    }

}
