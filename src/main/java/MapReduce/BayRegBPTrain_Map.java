package MapReduce;

import Jampack.*;
import NeuralNetwork.ANN_Train;
import NeuralNetwork.ArtificialNeuralNetwork;
import org.apache.commons.math3.analysis.function.Sin;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.Vector;

/**
 * Created by mlx on 4/25/16.
 */
public class BayRegBPTrain_Map extends
        Mapper<LongWritable, Text, Text, DoubleWritable> {
    private Text Windex = new Text();
    private Text Bindex = new Text();
    private String ANN_path = "";
    private double MSE_upperbound = 0.001;

    protected void getANNPath(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        this.ANN_path = conf.get("ThisIterationPath");
    }

    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        getANNPath(context);

        //pure text to String
        String line = value.toString();
        // Transform Input Data into wanted formation
        String[] LineDataArr = line.split("\n");
        Vector<Double[]> InputDataVec=new Vector<Double[]>();
        for(int t=0;t<LineDataArr.length;t++) {
            String[] SingleDataEntry = LineDataArr[t].split("\t");
            Double[] OneEntry = new Double[SingleDataEntry.length];
            for (int p = 0; p < SingleDataEntry.length; p++) {
                OneEntry[p] = Double.parseDouble(SingleDataEntry[p]);
            }
            InputDataVec.add(OneEntry);
        }
        //establish a ANN from existing file
        ArtificialNeuralNetwork TrainingANN = new ArtificialNeuralNetwork(ANN_path);
        ArtificialNeuralNetwork TotalUpdates= ANN_Train.BayReg_BPTrain(InputDataVec,TrainingANN,MSE_upperbound);

        context.write(new Text("SquareError"), new DoubleWritable(TotalUpdates.MSEofCertainSet));
        for (int i = 0; i < TotalUpdates.getANN().length; i++) {
            for (int j = 0; j < TotalUpdates.getANN()[i].getNeuronNum(); j++) {
                for (int k = 0; k < TotalUpdates.getANN()[i].getInputNum(); k++) {
                    String WeightIndex = "W-" + String.valueOf(i) + "-";
                    WeightIndex += String.valueOf(j) + "-" + String.valueOf(k);
                    Windex.set(WeightIndex);
                    context.write(Windex, new DoubleWritable(TotalUpdates.getANN()[i].getCertainWeight(j, k)));
                }
            }
        }
        for (int i = 0; i < TotalUpdates.getANN().length; i++) {
            for (int j = 0; j < TotalUpdates.getANN()[i].getNeuronNum(); j++) {
                String BiasIndex = "B-" + String.valueOf(i) + "-";
                BiasIndex += String.valueOf(j);
                Bindex.set(BiasIndex);
                context.write(Bindex, new DoubleWritable(TotalUpdates.getANN()[i].getCertainBias(j)));
            }
        }

    }
}
