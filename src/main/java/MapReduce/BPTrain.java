package MapReduce;

import HDFS_IO.ReadNWrite;
import NeuralNetwork.ArtificialNeuralNetwork;
import NeuralNetwork.NeuronLayer;
import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils;
import org.apache.commons.math3.analysis.function.Max;
import org.apache.hadoop.conf.Configuration;
import MapReduce.BPTrain_Map;
import MapReduce.BPTrain_Reduce;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hdfs.util.ByteArrayManager;
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
import org.apache.hadoop.util.Time;
import org.apache.hadoop.util.hash.Hash;
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.net.URLEncoder;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Created by mlx on 3/14/16.
 */
public class BPTrain {

    private static Job getValidationJob(Configuration conf, String Time, String Input, String Output) throws Exception {
        Job job = new Job(conf);
        job.setJarByClass(BPTrain.class);
        job.setJobName("Validation~" + Time);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        job.setMapperClass(ValidationMap.class);
        job.setReducerClass(ValidationReduce.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(Input));
        FileOutputFormat.setOutputPath(job, new Path(Output));

        job.setNumReduceTasks(1);

        return job;
    }

    private static Job getBayRegBP_SettedJob(Configuration conf, int IterationNum, String Time, String Input, String Output) throws Exception {
        Job job = new Job(conf);
        job.setJarByClass(BPTrain.class);
        job.setJobName("BayRegBPTrain" + String.valueOf(IterationNum) + "~" + Time);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        job.setMapperClass(BayRegBPTrain_Map.class);
        job.setReducerClass(BayRegBPTrain_Reduce.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(Input));
        FileOutputFormat.setOutputPath(job, new Path(Output));

        job.setNumReduceTasks(1);

        return job;
    }

    private static Job getLMBP_SettedJob(Configuration conf, int IterationNum, String Time, String Input, String Output) throws Exception {
        Job job = new Job(conf);
        job.setJarByClass(BPTrain.class);
        job.setJobName("LMBPTrain" + String.valueOf(IterationNum) + "~" + Time);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        job.setMapperClass(LMBPTrain_Map.class);
        job.setReducerClass(LMBPTrain_Reduce.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(Input));
        FileOutputFormat.setOutputPath(job, new Path(Output));

        job.setNumReduceTasks(5);

        return job;
    }

    private static Job getCGBP_SettedJob(Configuration conf, int IterationNum, String Time, String Input, String Output) throws Exception {
        Job job = new Job(conf);
        job.setJarByClass(BPTrain.class);
        job.setJobName("CGBPTrain" + String.valueOf(IterationNum) + "~" + Time);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        job.setMapperClass(CGBPTrain_Map.class);
        job.setReducerClass(CGBPTrain_Reduce.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(Input));
        FileOutputFormat.setOutputPath(job, new Path(Output));

        job.setNumReduceTasks(5);

        return job;
    }

    private static Job getSDBP_SettedJob(Configuration conf, int IterationNum, String Time, String Input, String Output) throws Exception {
        Job job = new Job(conf);
        job.setJarByClass(BPTrain.class);
        job.setJobName("SDBPTrain" + String.valueOf(IterationNum) + " " + Time);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        job.setMapperClass(BPTrain_Map.class);
        job.setReducerClass(BPTrain_Reduce.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(Input));
        FileOutputFormat.setOutputPath(job, new Path(Output));

        job.setNumReduceTasks(5);

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

        return WeightMap;
    }

    public static Double getMSEFromFile(String Path) throws IOException {
        Double SE = 100.0;
        int partNum = 0;
        String partPath = Path + "/part-r-" + fullStringNum(partNum);
        while (ReadNWrite.hdfs_isFileExist(partPath)) {
            Vector WeightChangeArr = ReadNWrite.hdfs_Read(partPath);
            boolean isFind = false;
            for (int i = 0; i < WeightChangeArr.size(); i++) {
                String[] IndexNChange = ((String) WeightChangeArr.get(i)).split("\t");
                if (IndexNChange[0].equals("MeanSquareError")) {
                    SE = Double.parseDouble(IndexNChange[1]);
                    isFind = true;
                    break;
                }
            }
            if (isFind) {
                break;
            }
            partNum++;
            partPath = Path + "/part-r-" + fullStringNum(partNum);
        }
        return SE;
    }

    public static void discardCertainUpdateForANN(String oldANNPath, String WeightChangePath, String SavePath) throws IOException {
        ArtificialNeuralNetwork oldANN = new ArtificialNeuralNetwork(oldANNPath);
        Map<String, Double> WeightHash_now = getWeightUpdatesFromFile(WeightChangePath);

        Iterator iter1 = WeightHash_now.entrySet().iterator();
        while (iter1.hasNext()) {
            Map.Entry entry = (Map.Entry) iter1.next();
            String key = (String) entry.getKey();
            Double value = -((Double) entry.getValue());

            String[] IndexArr = key.split("-");
            if (IndexArr.length == 4) {
                int LayNum = Integer.parseInt(IndexArr[1]);
                int NeuronNum = Integer.parseInt(IndexArr[2]);
                int WeightNum = Integer.parseInt(IndexArr[3]);

                oldANN.updateCertainWeight(LayNum, NeuronNum, WeightNum, value);
            } else if (IndexArr.length == 3) {
                int LayNum = Integer.parseInt(IndexArr[1]);
                int NeuronNum = Integer.parseInt(IndexArr[2]);

                oldANN.updateCertainBias(LayNum, NeuronNum, value);
            }
        }
        String[] new_ANN_content = oldANN.saveANN();
        ReadNWrite.hdfs_Write(new_ANN_content, SavePath);

    }

    public static void getNewANN_Usual(String oldANNPath, String WeightChangePath, String SavePath) throws IOException {
        ArtificialNeuralNetwork oldANN = new ArtificialNeuralNetwork(oldANNPath);

        Map<String, Double> WeightHash_now = getWeightUpdatesFromFile(WeightChangePath);

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

                oldANN.updateCertainWeight(LayNum, NeuronNum, WeightNum, value);
            } else if (IndexArr.length == 3) {
                int LayNum = Integer.parseInt(IndexArr[1]);
                int NeuronNum = Integer.parseInt(IndexArr[2]);

                oldANN.updateCertainBias(LayNum, NeuronNum, value);
            }
        }

        String[] new_ANN_content = oldANN.saveANN();
        ReadNWrite.hdfs_Write(new_ANN_content, SavePath);
//        double lastGenerationErrorSum = 0.0;
//        int partNum = 0;
//        String partPath = WeightChangePath + "/part-r-" + fullStringNum(partNum);
//        while (ReadNWrite.hdfs_isFileExist(partPath)) {
//            Vector WeightChangeArr = ReadNWrite.hdfs_Read(partPath);
//            for (int i = 0; i < WeightChangeArr.size(); i++) {
//                String[] IndexNChange = ((String) WeightChangeArr.get(i)).split("\t");
//                double ChangeValue = Double.parseDouble(IndexNChange[1]);
//
//                String[] IndexArr = IndexNChange[0].split("-");
//                if (IndexArr.length == 4) {
//                    int LayNum = Integer.parseInt(IndexArr[1]);
//                    int NeuronNum = Integer.parseInt(IndexArr[2]);
//                    int WeightNum = Integer.parseInt(IndexArr[3]);
//                    oldANN.updateCertainWeight(LayNum, NeuronNum, WeightNum, ChangeValue);
//                } else if (IndexArr.length == 3) {
//                    int LayNum = Integer.parseInt(IndexArr[1]);
//                    int NeuronNum = Integer.parseInt(IndexArr[2]);
//                    oldANN.updateCertainBias(LayNum, NeuronNum, ChangeValue);
//                } else {
//                    lastGenerationErrorSum += ChangeValue;
//                }
//            }
//            partNum++;
//            partPath = WeightChangePath + "/part-r-" + fullStringNum(partNum);
//        }
//
//
//        String[] new_ANN_content = oldANN.saveANN();
//        ReadNWrite.hdfs_Write(new_ANN_content, SavePath);
//        return lastGenerationErrorSum;
    }

    public static void getNewANN_Momentum(String oldANNPath, String WeightChangePath, String SavePath, String LastWeightPath, double Momentum) throws IOException {
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
            }
        }

        String[] new_ANN_content = oldANN.saveANN();
        ReadNWrite.hdfs_Write(new_ANN_content, SavePath);
    }

    private static void run_SDBP_Usual(ArtificialNeuralNetwork InitialANN, String ipPrefix, String InputPath, String OutputPath, int MaxIterationNum) throws Exception {
        String[] pathArr = OutputPath.split("/");
        String pathPrefix = "";
        for (int i = 1; i < pathArr.length - 1; i++) {
            pathPrefix += "/" + pathArr[i];
        }
        for (int IterationNum = 0; IterationNum < MaxIterationNum; IterationNum++) {
            Configuration conf = new Configuration();

            SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd_HH:mm:ss");//设置日期格式
            String TimeNow = df.format(new Date());

            if (IterationNum == 0) {
                ReadNWrite.hdfs_Write(InitialANN.saveANN(), ipPrefix + pathPrefix + "/result_ANN-0");
                conf.set("ThisIterationPath", ipPrefix + pathPrefix + "/result_ANN-0");
            } else {
                String oldANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum - 1);
                String newANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum);
                String ChangeValuePath = ipPrefix + OutputPath + "-" + String.valueOf(IterationNum - 1);

                BPTrain.getNewANN_Usual(oldANNPath, ChangeValuePath, newANNPath);
                conf.set("ThisIterationPath", newANNPath);
            }
            //Set job configs
            String outPath = OutputPath + "-" + String.valueOf(IterationNum);
            Job job = BPTrain.getSDBP_SettedJob(conf, IterationNum, TimeNow, InputPath, outPath);
            //hand in the job
            job.waitForCompletion(true);
        }
    }

    private static void run_SDBP_Momentum(ArtificialNeuralNetwork InitialANN, String ipPrefix, String InputPath, String OutputPath, int MaxIterationNum, double Momentum) throws Exception {
        String[] pathArr = OutputPath.split("/");
        String pathPrefix = "";
        for (int i = 1; i < pathArr.length - 1; i++) {
            pathPrefix += "/" + pathArr[i];
        }
        for (int IterationNum = 0; IterationNum < MaxIterationNum; IterationNum++) {
            Configuration conf = new Configuration();

            SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd_HH:mm:ss");//设置日期格式
            String TimeNow = df.format(new Date());

            if (IterationNum >= 2) {
                String oldANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum - 1);
                String newANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum);
                String ChangeValuePath = ipPrefix + OutputPath + "-" + String.valueOf(IterationNum - 1);
                String LastChangePath = ipPrefix + OutputPath + "-" + String.valueOf(IterationNum - 2);

                BPTrain.getNewANN_Momentum(oldANNPath, ChangeValuePath, newANNPath, LastChangePath, Momentum);
                conf.set("ThisIterationPath", newANNPath);
            } else if (IterationNum == 0) {
                ReadNWrite.hdfs_Write(InitialANN.saveANN(), ipPrefix + pathPrefix + "/result_ANN-0");
                conf.set("ThisIterationPath", ipPrefix + pathPrefix + "/result_ANN-0");
            } else {
                String oldANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum - 1);
                String newANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum);
                String ChangeValuePath = ipPrefix + OutputPath + "-" + String.valueOf(IterationNum - 1);

                BPTrain.getNewANN_Usual(oldANNPath, ChangeValuePath, newANNPath);
                conf.set("ThisIterationPath", newANNPath);
            }
            //Set job configs
            String outPath = OutputPath + "-" + String.valueOf(IterationNum);
            Job job = getSDBP_SettedJob(conf, IterationNum, TimeNow, InputPath, outPath);

            job.waitForCompletion(true);
        }
    }

    //parameter "eta" controls the multiplying ratio for LearningRate to increase when the MSE decreases at one iteration;
    //paremeter "rho" controls the multiplying ratio for LearningRate to decrease when the MSE increases more than "ksi" times at one iteration;
    //paramete  "ksi" controls the upper bound ratio between two iteration MSE.If MSE of this iteration is more than "ksi" times than last iteration,we should take special measures;
    private static void run_SDBP_MomentumVL(ArtificialNeuralNetwork InitialANN, String ipPrefix, String InputPath, String OutputPath, int MaxIterationNum, double Momentum, double eta, double rho, double ksi) throws Exception {
        String[] pathArr = OutputPath.split("/");
        String pathPrefix = "";
        for (int i = 1; i < pathArr.length - 1; i++) {
            pathPrefix += "/" + pathArr[i];
        }
        double thisIteSE = 0.0;
        double lastIteSE = 0.0;

        double VaryingLearningRate = 0.2;
        double VaryingMomentum;

        double[] ValidationMSE=new double[MaxIterationNum];

        for (int IterationNum = 0; IterationNum < MaxIterationNum; IterationNum++, lastIteSE = thisIteSE) {
            Configuration conf = new Configuration();

            SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd_HH:mm:ss");//设置日期格式
            String TimeNow = df.format(new Date());

            if (IterationNum >= 2) {
                String oldANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum - 1);
                String newANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum);
                String ChangeValuePath = ipPrefix + OutputPath + "-" + String.valueOf(IterationNum - 1);

                thisIteSE = BPTrain.getMSEFromFile(ChangeValuePath);

                double SE_VaryingRate = ((thisIteSE - lastIteSE) / (lastIteSE));
                if (SE_VaryingRate < 0) {
                    String LastChangePath = ipPrefix + OutputPath + "-" + String.valueOf(IterationNum - 2);
                    VaryingMomentum = Momentum;

                    BPTrain.getNewANN_Usual(oldANNPath, ChangeValuePath, newANNPath);
                    conf.set("ThisIterationPath", newANNPath);
                    VaryingLearningRate *= eta;
                    conf.set("LearningRate", String.valueOf(VaryingLearningRate));
                    conf.set("LastUpdatePath", LastChangePath);
                    conf.set("Momentum", String.valueOf(VaryingMomentum));

                } else if (0 <= SE_VaryingRate && SE_VaryingRate <= ksi) {
                    String LastChangePath = ipPrefix + OutputPath + "-" + String.valueOf(IterationNum - 2);
                    VaryingMomentum = Momentum;

                    BPTrain.getNewANN_Usual(oldANNPath, ChangeValuePath, newANNPath);
                    conf.set("ThisIterationPath", newANNPath);
                    conf.set("LearningRate", String.valueOf(VaryingLearningRate));
                    conf.set("LastUpdatePath", LastChangePath);
                    conf.set("Momentum", String.valueOf(VaryingMomentum));
                } else {
                    String DiscardedUpdatePath = ipPrefix + OutputPath + "-" + String.valueOf(IterationNum - 2);
                    BPTrain.discardCertainUpdateForANN(oldANNPath, DiscardedUpdatePath, newANNPath);
                    conf.set("ThisIterationPath", newANNPath);
                    VaryingLearningRate *= rho;
                    conf.set("LearningRate", String.valueOf(VaryingLearningRate));
                }

            } else if (IterationNum == 0) {
                ReadNWrite.hdfs_Write(InitialANN.saveANN(), ipPrefix + pathPrefix + "/result_ANN-0");
                conf.set("ThisIterationPath", ipPrefix + pathPrefix + "/result_ANN-0");
                conf.set("LearningRate", String.valueOf(VaryingLearningRate));
            } else {
                String oldANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum - 1);
                String newANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum);
                String ChangeValuePath = ipPrefix + OutputPath + "-" + String.valueOf(IterationNum - 1);

                thisIteSE = BPTrain.getMSEFromFile(ChangeValuePath); //ERR0

                BPTrain.getNewANN_Usual(oldANNPath, ChangeValuePath, newANNPath);

                conf.set("ThisIterationPath", newANNPath);
                conf.set("LearningRate", String.valueOf(VaryingLearningRate));
            }
            String ValidationSetPath = "hdfs://Master:9000/user/mlx/ValidationSet";
            String ValidationOutPath = OutputPath + "-Validation-" + String.valueOf(IterationNum);
            Job ValiJob = getValidationJob(conf, TimeNow, ValidationSetPath, ValidationOutPath);
            ValiJob.waitForCompletion(true);
            ValidationMSE[IterationNum] = BPTrain.getMSEFromFile(ValidationOutPath);
            if (IterationNum >= 3) {
                if (ValidationMSE[IterationNum] > ValidationMSE[IterationNum - 1] && ValidationMSE[IterationNum] > ValidationMSE[IterationNum - 2] && ValidationMSE[IterationNum] > ValidationMSE[IterationNum - 3]) {
                    break;
                }
            }

            //Set job configs
            String outPath = OutputPath + "-" + String.valueOf(IterationNum);
            Job job = getSDBP_SettedJob(conf, IterationNum, TimeNow, InputPath, outPath);
            job.waitForCompletion(true);
        }
    }

    private static void run_CGBP(ArtificialNeuralNetwork InitialANN, String ipPrefix, String InputPath, String OutputPath, int MaxIterationNum) throws Exception {
        String[] pathArr = OutputPath.split("/");
        String pathPrefix = "";
        for (int i = 1; i < pathArr.length - 1; i++) {
            pathPrefix += "/" + pathArr[i];
        }
        for (int IterationNum = 0; IterationNum < MaxIterationNum; IterationNum++) {
            Configuration conf = new Configuration();

            SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd_HH:mm:ss");//设置日期格式
            String TimeNow = df.format(new Date());

            if (IterationNum == 0) {
                ReadNWrite.hdfs_Write(InitialANN.saveANN(), ipPrefix + pathPrefix + "/result_ANN-0");
                conf.set("ThisIterationPath", ipPrefix + pathPrefix + "/result_ANN-0");
            } else {
                String oldANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum - 1);
                String newANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum);
                String ChangeValuePath = ipPrefix + OutputPath + "-" + String.valueOf(IterationNum - 1);

                BPTrain.getNewANN_Usual(oldANNPath, ChangeValuePath, newANNPath);
                conf.set("ThisIterationPath", newANNPath);
            }
            //Set job configs
            String outPath = OutputPath + "-" + String.valueOf(IterationNum);
            Job job = BPTrain.getCGBP_SettedJob(conf, IterationNum, TimeNow, InputPath, outPath);
            //hand in the job
            job.waitForCompletion(true);
        }

    }

    private static void run_LMBP(ArtificialNeuralNetwork InitialANN, String ipPrefix, String InputPath, String OutputPath, int MaxIterationNum) throws Exception {
        String[] pathArr = OutputPath.split("/");
        String pathPrefix = "";
        for (int i = 1; i < pathArr.length - 1; i++) {
            pathPrefix += "/" + pathArr[i];
        }
        for (int IterationNum = 0; IterationNum < MaxIterationNum; IterationNum++) {
            Configuration conf = new Configuration();

            SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd_HH:mm:ss");//设置日期格式
            String TimeNow = df.format(new Date());

            if (IterationNum == 0) {
                ReadNWrite.hdfs_Write(InitialANN.saveANN(), ipPrefix + pathPrefix + "/result_ANN-0");
                conf.set("ThisIterationPath", ipPrefix + pathPrefix + "/result_ANN-0");
            } else {
                String oldANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum - 1);
                String newANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum);
                String ChangeValuePath = ipPrefix + OutputPath + "-" + String.valueOf(IterationNum - 1);

                BPTrain.getNewANN_Usual(oldANNPath, ChangeValuePath, newANNPath);
                conf.set("ThisIterationPath", newANNPath);
            }
            //Set job configs
            String outPath = OutputPath + "-" + String.valueOf(IterationNum);
            Job job = BPTrain.getLMBP_SettedJob(conf, IterationNum, TimeNow, InputPath, outPath);
            //hand in the job
            job.waitForCompletion(true);
        }
    }

    private static void run_BayRegBPTrain(ArtificialNeuralNetwork InitialANN, String ipPrefix, String InputPath, String OutputPath, int MaxIterationNum) throws Exception{
        String[] pathArr = OutputPath.split("/");
        String pathPrefix = "";
        for (int i = 1; i < pathArr.length - 1; i++) {
            pathPrefix += "/" + pathArr[i];
        }
        for (int IterationNum = 0; IterationNum < MaxIterationNum; IterationNum++) {
            Configuration conf = new Configuration();

            SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd_HH:mm:ss");//设置日期格式
            String TimeNow = df.format(new Date());

            if (IterationNum == 0) {
                ReadNWrite.hdfs_Write(InitialANN.saveANN(), ipPrefix + pathPrefix + "/result_ANN-0");
                conf.set("ThisIterationPath", ipPrefix + pathPrefix + "/result_ANN-0");
            } else {
                String oldANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum - 1);
                String newANNPath = ipPrefix + pathPrefix + "/result_ANN-" + String.valueOf(IterationNum);
                String ChangeValuePath = ipPrefix + OutputPath + "-" + String.valueOf(IterationNum - 1);

                BPTrain.getNewANN_Usual(oldANNPath, ChangeValuePath, newANNPath);
                conf.set("ThisIterationPath", newANNPath);
            }
            //Set job configs
            String outPath = OutputPath + "-" + String.valueOf(IterationNum);
            Job job = BPTrain.getBayRegBP_SettedJob(conf, IterationNum, TimeNow, InputPath, outPath);
            //hand in the job
            job.waitForCompletion(true);
        }

    }

    public static void main(String[] args) throws Exception {
        String ipPrefix = "hdfs://Master:9000";

        int RunFucCode = Integer.parseInt(args[2]);
        int TotalIterationNum = Integer.parseInt(args[3]);

//        int InputNum = 41;
//        int LayerNum = 3;
//        int[] NumEachLayer = {7, 10, 1};
//        int[] IndexEachLayer = {1, 4, 4};
        int InputNum = 1;
        int LayerNum = 2;
        int[] NumEachLayer = {2,1};
        int[] IndexEachLayer = {1, 3};

        ArtificialNeuralNetwork InitialANN = new ArtificialNeuralNetwork(InputNum, LayerNum, NumEachLayer, IndexEachLayer);

        if (RunFucCode == 0) {
            BPTrain.run_SDBP_Usual(InitialANN, ipPrefix, args[0], args[1], TotalIterationNum);
        } else if (RunFucCode == 1) {
            BPTrain.run_SDBP_Momentum(InitialANN, ipPrefix, args[0], args[1], TotalIterationNum, 0.8);
        } else if (RunFucCode == 2) {
            BPTrain.run_SDBP_MomentumVL(InitialANN, ipPrefix, args[0], args[1], TotalIterationNum, 0.8, 1.05, 0.7, 0.04);
        } else if (RunFucCode == 3) {
            BPTrain.run_CGBP(InitialANN, ipPrefix, args[0], args[1], TotalIterationNum);
        } else if (RunFucCode == 4 ){
            BPTrain.run_LMBP(InitialANN, ipPrefix, args[0], args[1], TotalIterationNum);
        } else{
            BPTrain.run_BayRegBPTrain(InitialANN, ipPrefix, args[0], args[1], TotalIterationNum);
        }
        //BPTrain.run_SDBP_MomentumVL(InitialANN, ipPrefix, args[0], args[1], TotalIterationNum, 0.8, 1.05, 0.7, 0.04);
        //BPTrain.run_CGBP(InitialANN, ipPrefix, args[0], args[1], TotalIterationNum);

//        for (; IterationNum < TotalIterationNum; IterationNum++) {
//            Configuration conf = new Configuration();
//
//            if (IterationNum >= 2) {
//                String oldANNPath = ipPrefix + pathPrefix + "/testANN" + String.valueOf(IterationNum - 1);
//                String newANNPath = ipPrefix + pathPrefix + "/testANN" + String.valueOf(IterationNum);
//                String ChangeValuePath = ipPrefix + args[1] + "-" + String.valueOf(IterationNum - 1);
//                String LastChangePath = ipPrefix + args[1] + "-" + String.valueOf(IterationNum - 2);
//
//                ThisIterationError = BPTrain.getNewANN_Momentum(oldANNPath, ChangeValuePath, newANNPath, LastChangePath, 0.8);
//                conf.set("ThisIterationPath", newANNPath);
//
//            } else if (IterationNum == 0) {
//                int InputNum = 1;
//                int LayerNum = 2;
//                int[] NumEachLayer = {2, 1};
//                int[] IndexEachLayer = {1, 3};
//
//                ArtificialNeuralNetwork InitialANN = new ArtificialNeuralNetwork(InputNum, LayerNum, NumEachLayer, IndexEachLayer);
//                ReadNWrite.hdfs_Write(InitialANN.saveANN(), ipPrefix + pathPrefix + "/testANN0");
//                conf.set("ThisIterationPath", ipPrefix + pathPrefix + "/testANN0");
//            } else {
//                String oldANNPath = ipPrefix + pathPrefix + "/testANN" + String.valueOf(IterationNum - 1);
//                String newANNPath = ipPrefix + pathPrefix + "/testANN" + String.valueOf(IterationNum);
//                String ChangeValuePath = ipPrefix + args[1] + "-" + String.valueOf(IterationNum - 1);
//
//                ThisIterationError = BPTrain.getNewANN_Usual(oldANNPath, ChangeValuePath, newANNPath);
//                conf.set("ThisIterationPath", newANNPath);
//            }
//
//            //Set job configs
//            Job job = getSettedJob(conf, IterationNum);
//            FileInputFormat.addInputPath(job, new Path(args[0]));
//
//            String outPath = args[1] + "-" + String.valueOf(IterationNum);
//            FileOutputFormat.setOutputPath(job, new Path(outPath));
//
//            job.waitForCompletion(true);
//
//            LastIterationError = ThisIterationError;
//        }
    }

}
