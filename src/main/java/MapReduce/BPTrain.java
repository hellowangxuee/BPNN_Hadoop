package MapReduce;

import HDFS_IO.ReadNWrite;
import NeuralNetwork.ArtificialNeuralNetwork;
import NeuralNetwork.NeuronLayer;
import org.apache.hadoop.conf.Configuration;
import MapReduce.BPTrain_Map;
import MapReduce.BPTrain_Reduce;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hdfs.util.EnumCounters;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.jobhistory.JobHistoryParser;
import org.apache.hadoop.mapreduce.lib.aggregate.ValueAggregator;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.v2.app.webapp.dao.JobInfo;
import org.apache.hadoop.util.hash.Hash;
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.net.URLEncoder;
import java.util.*;

/**
 * Created by mlx on 3/14/16.
 */
public class BPTrain {

    private static Job getSettedJob(Configuration conf, int IterationNum) throws Exception {
        Job job = new Job(conf);
        job.setJarByClass(BPTrain.class);
        job.setJobName("BPTrain" + String.valueOf(IterationNum));

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        job.setMapperClass(BPTrain_Map.class);
        job.setReducerClass(BPTrain_Reduce.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        job.setNumReduceTasks(3);

        return job;
    }

    private static String fullStringNum(int n) {
        if (0 <= n && n <= 9) {
            String fullStr = "0000" + String.valueOf(n);
            return fullStr;
        } else if (n <= 99) {
            String fullStr = "000" + String.valueOf(n);
            return fullStr;
        } else if (n <= 999) {
            String fullStr = "00" + String.valueOf(n);
            return fullStr;
        } else if (n <= 9999) {
            String fullStr = "0" + String.valueOf(n);
            return fullStr;
        } else {
            String fullStr = String.valueOf(n);
            return fullStr;
        }
    }

    public static HashMap<String, Double> getWeightUpdatesFromFile(String WeightUpdatesPath) throws IOException {
        HashMap<String, Double> WeightMap = new HashMap<String, Double>();

        int partNum = 0;
        String partPath = WeightUpdatesPath + "/part-r-" + fullStringNum(partNum);
        while (ReadNWrite.hdfs_isFileExist(partPath)) {
            Vector WeightChangeArr = ReadNWrite.hdfs_Read(partPath);
            for (int i = 0; i < WeightChangeArr.size(); i++) {
                String[] IndexNChange = ((String) WeightChangeArr.get(i)).split("\t");
                double ChangeValue = Double.parseDouble(IndexNChange[1]);

                if (WeightMap.containsKey(IndexNChange[0])) {
                    Double newValue = WeightMap.get(IndexNChange[0]) + ChangeValue;
                    WeightMap.put(IndexNChange[0], newValue);
                } else {
                    WeightMap.put(IndexNChange[0], ChangeValue);
                }

            }
            partNum++;
            partPath = WeightUpdatesPath + "/part-r-" + fullStringNum(partNum);
        }
//        WeightMap.put("PartNum", (double) partNum);

        return WeightMap;
    }

    public static double generateNewANN(String oldANNPath, String WeightChangePath, String SavePath) throws IOException {
        ArtificialNeuralNetwork oldANN = new ArtificialNeuralNetwork(oldANNPath);

        double lastGenerationErrorSum = 0.0;
        int partNum = 0;
        String partPath = WeightChangePath + "/part-r-" + fullStringNum(partNum);
        while (ReadNWrite.hdfs_isFileExist(partPath)) {
            Vector WeightChangeArr = ReadNWrite.hdfs_Read(partPath);
            for (int i = 0; i < WeightChangeArr.size(); i++) {
                String[] IndexNChange = ((String) WeightChangeArr.get(i)).split("\t");
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
                    lastGenerationErrorSum += ChangeValue;
                }
            }
            partNum++;
            partPath = WeightChangePath + "/part-r-" + fullStringNum(partNum);
        }


        String[] new_ANN_content = oldANN.saveANN();
        ReadNWrite.hdfs_Write(new_ANN_content, SavePath);
        return lastGenerationErrorSum;
    }

//    public static double getNewANN_Ave(String oldANNPath, String WeightChangePath, String SavePath) throws IOException {
//        double lastGenerationErrorSum = 0.0;
//        ArtificialNeuralNetwork oldANN = new ArtificialNeuralNetwork(oldANNPath);
//
//        Map<String, Double> WeightHash = getWeightUpdatesFromFile(WeightChangePath);
//        Double partNum = WeightHash.get("PartNum");
////        double lastGenerationErrorSum = 0.0;
////        int partNum = 0;
////        String partPath = WeightChangePath + "/part-r-" + fullStringNum(partNum);
////        while (ReadNWrite.hdfs_isFileExist(partPath)) {
////            Vector WeightChangeArr = ReadNWrite.hdfs_Read(partPath);
////            for (int i = 0; i < WeightChangeArr.size(); i++) {
////                String[] IndexNChange = ((String) WeightChangeArr.get(i)).split("\t");
////                double ChangeValue = Double.parseDouble(IndexNChange[1]);
////
////                if (IndexNChange[0].contains("W")) {
////                    if (WeightHash.containsKey(IndexNChange[0])) {
////                        Double newValue = WeightHash.get(IndexNChange[0]) + ChangeValue;
////                        WeightHash.put(IndexNChange[0], newValue);
////                    } else {
////                        WeightHash.put(IndexNChange[0], ChangeValue);
////                    }
////                } else if (IndexNChange[0].contains("B")) {
////                    if (BiasHash.containsKey(IndexNChange[0])) {
////                        Double newValue = BiasHash.get(IndexNChange[0]) + ChangeValue;
////                        BiasHash.put(IndexNChange[0], newValue);
////                    } else {
////                        BiasHash.put(IndexNChange[0], ChangeValue);
////                    }
////                } else {
////                    lastGenerationErrorSum += ChangeValue;
////                }
////
////            }
////            partNum++;
////            partPath = WeightChangePath + "/part-r-" + fullStringNum(partNum);
////        }
//
//        Iterator iter1 = WeightHash.entrySet().iterator();
//        while (iter1.hasNext()) {
//            Map.Entry entry = (Map.Entry) iter1.next();
//            String key = (String) entry.getKey();
//            if (key.equals("PartNum")) {
//                continue;
//            } else {
//                Double value = (Double) entry.getValue();
//                String[] IndexArr = key.split("-");
//                if (IndexArr.length == 4) {
//                    int LayNum = Integer.parseInt(IndexArr[1]);
//                    int NeuronNum = Integer.parseInt(IndexArr[2]);
//                    int WeightNum = Integer.parseInt(IndexArr[3]);
//                    oldANN.updateCertainWeight(LayNum, NeuronNum, WeightNum, (value / partNum));
//                } else if (IndexArr.length == 3) {
//                    int LayNum = Integer.parseInt(IndexArr[1]);
//                    int NeuronNum = Integer.parseInt(IndexArr[2]);
//                    oldANN.updateCertainBias(LayNum, NeuronNum, (value / partNum));
//                } else {
//                    lastGenerationErrorSum = value;
//                }
//            }
//        }
//
//        String[] new_ANN_content = oldANN.saveANN();
//        ReadNWrite.hdfs_Write(new_ANN_content, SavePath);
//        return lastGenerationErrorSum;
//    }

    public static double getNewANN_Momentum(String oldANNPath, String WeightChangePath, String SavePath, String LastWeightPath, double Momentum) throws IOException {
        double lastGenerationErrorSum = 0.0;
        ArtificialNeuralNetwork oldANN = new ArtificialNeuralNetwork(oldANNPath);

        Map<String, Double> WeightHash_now = getWeightUpdatesFromFile(WeightChangePath);
        Map<String, Double> WeightHash_last = getWeightUpdatesFromFile(LastWeightPath);

        Iterator iter1 = WeightHash_now.entrySet().iterator();
        while (iter1.hasNext()) {
            Map.Entry entry = (Map.Entry) iter1.next();
            String key = (String) entry.getKey();
            Double value = (Double) entry.getValue();

            String[] IndexArr = key.split("-");
            if (IndexArr.length == 4) {
                int LayNum = Integer.parseInt(IndexArr[1]);
                int NeuronNum = Integer.parseInt(IndexArr[2]);
                int WeightNum = Integer.parseInt(IndexArr[3]);

                Double value_last = WeightHash_last.get(key);
                Double Momentum_value = Momentum * (value_last) + (1 - Momentum) * (value);

                oldANN.updateCertainWeight(LayNum, NeuronNum, WeightNum, Momentum_value);
            } else if (IndexArr.length == 3) {
                int LayNum = Integer.parseInt(IndexArr[1]);
                int NeuronNum = Integer.parseInt(IndexArr[2]);

                Double value_last = WeightHash_last.get(key);
                Double Momentum_value = Momentum * (value_last) + (1 - Momentum) * (value);

                oldANN.updateCertainBias(LayNum, NeuronNum, Momentum_value);
            } else {
                lastGenerationErrorSum = value;
            }
        }

        String[] new_ANN_content = oldANN.saveANN();
        ReadNWrite.hdfs_Write(new_ANN_content, SavePath);
        return lastGenerationErrorSum;
    }

    public static void main(String[] args) throws Exception {
        int IterationNum = 0;
        int TotalIterationNum = 150;
        String ipPrefix = "hdfs://Master:9000";

        String[] pathArr = args[1].split("/");
        String pathPrefix = "";
        for (int i = 1; i < pathArr.length - 1; i++) {
            pathPrefix += "/" + pathArr[i];
        }

        double ThisIterationError = 0.0;
        double LastIterationError = 0.0;

        for (; IterationNum < TotalIterationNum; IterationNum++) {
            Configuration conf = new Configuration();

            if (IterationNum >= 2) {
                String oldANNPath = ipPrefix + pathPrefix + "/testANN" + String.valueOf(IterationNum - 1);
                String newANNPath = ipPrefix + pathPrefix + "/testANN" + String.valueOf(IterationNum);
                String ChangeValuePath = ipPrefix + args[1] + "-" + String.valueOf(IterationNum - 1);
                String LastChangePath = ipPrefix + args[1] + "-" + String.valueOf(IterationNum - 2);

                ThisIterationError = BPTrain.getNewANN_Momentum(oldANNPath, ChangeValuePath, newANNPath, LastChangePath, 0.8);
                conf.set("ThisIterationPath", newANNPath);

            } else if (IterationNum == 0) {
                int InputNum = 1;
                int LayerNum = 2;
                int[] NumEachLayer = {2, 1};
                int[] IndexEachLayer = {1, 3};

                ArtificialNeuralNetwork InitialANN = new ArtificialNeuralNetwork(InputNum, LayerNum, NumEachLayer, IndexEachLayer);
                ReadNWrite.hdfs_Write(InitialANN.saveANN(), ipPrefix + pathPrefix + "/testANN0");
                conf.set("ThisIterationPath", ipPrefix + pathPrefix + "/testANN0");
            } else {
                String oldANNPath = ipPrefix + pathPrefix + "/testANN" + String.valueOf(IterationNum - 1);
                String newANNPath = ipPrefix + pathPrefix + "/testANN" + String.valueOf(IterationNum);
                String ChangeValuePath = ipPrefix + args[1] + "-" + String.valueOf(IterationNum - 1);

                ThisIterationError = BPTrain.generateNewANN(oldANNPath, ChangeValuePath, newANNPath);
                conf.set("ThisIterationPath", newANNPath);
            }

            //Set job configs
            Job job = getSettedJob(conf, IterationNum);
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

            LastIterationError = ThisIterationError;
        }
    }

}
