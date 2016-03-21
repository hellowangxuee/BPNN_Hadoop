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

    private static String fullStringNum(int n){
        if(0<=n && n<=9){
            String fullStr="0000"+String.valueOf(n);
            return fullStr;
        }
        else if(n<=99){
            String fullStr="000"+String.valueOf(n);
            return fullStr;
        }
        else if(n<=999){
            String fullStr="00"+String.valueOf(n);
            return fullStr;
        }
        else if(n<=9999){
            String fullStr="0"+String.valueOf(n);
            return fullStr;
        }
        else{
            String fullStr=String.valueOf(n);
            return fullStr;
        }
    }

    public static void generateNewANN(String oldANNPath,String WeightChangePath,String SavePath) throws IOException{
        ArtificialNeuralNetwork oldANN=new ArtificialNeuralNetwork(oldANNPath);

        int partNum=0;
        String partPath=SavePath+"/part-r-"+fullStringNum(partNum);
        while(ReadNWrite.hdfs_isFileExist(partPath)) {
            String[] WeightChangeArr = ReadNWrite.hdfs_Read(WeightChangePath);
            for (int i = 0; i < WeightChangeArr.length; i++) {
                String[] IndexNChange = WeightChangeArr[i].split("\t");
                double ChangeValue = Double.parseDouble(IndexNChange[1]);

                String[] IndexArr = IndexNChange[0].split("-");
                if (IndexArr.length == 4) {
                    int LayNum = Integer.parseInt(IndexArr[1]);
                    int NeuronNum = Integer.parseInt(IndexArr[2]);
                    int WeightNum = Integer.parseInt(IndexArr[3]);
                    oldANN.updateCertainWeight(LayNum, NeuronNum, WeightNum, ChangeValue);
                } else if (IndexArr.length == 3) {
                    int LayNum = Integer.parseInt(IndexArr[1]);
                    int NeuronNum = Integer.parseInt(IndexArr[2]);
                    oldANN.updateCertainBias(LayNum, NeuronNum, ChangeValue);
                } else {
                    continue;
                }
            }
            partNum++;
            partPath=SavePath+"/part-r-"+fullStringNum(partNum);
        }

        String[] new_ANN_content=oldANN.saveANN();
        ReadNWrite.hdfs_Write(new_ANN_content,SavePath);

    }

    public static void main(String[] args) throws Exception {
        int IterationNum=0;
        int TotalIterationNum=100;
        String ipPrefix="hdfs://Master:9000";

        String[] pathArr=args[0].split("/");
        String pathPrefix="";
        for(int i=1;i<pathArr.length-1;i++) {
            pathPrefix += "/" + pathArr[i];
        }

        for(;IterationNum<TotalIterationNum;IterationNum++) {
            Configuration conf = new Configuration();
            if (IterationNum==0) {
                int InputNum=1;
                int LayerNum=2;
                int[] NumEachLayer={2,1};
                int[] IndexEachLayer={1,3};

                ArtificialNeuralNetwork InitialANN=new ArtificialNeuralNetwork(InputNum,LayerNum,NumEachLayer,IndexEachLayer);
                ReadNWrite.hdfs_Write(InitialANN.saveANN(),ipPrefix+pathPrefix+"/testANN0") ;
                conf.set("ThisIterationPath", ipPrefix+pathPrefix+"/testANN0");
            }
            else{
                String oldANNPath=ipPrefix+pathPrefix+"/testANN"+String.valueOf(IterationNum-1);
                String newANNPath=ipPrefix+pathPrefix+"/testANN"+String.valueOf(IterationNum);
                String ChangeValuePath=ipPrefix+args[1]+"-"+String.valueOf(IterationNum-1);

                BPTrain.generateNewANN(oldANNPath,ChangeValuePath,newANNPath);
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
