package MapReduce;

import NeuralNetwork.ArtificialNeuralNetwork;
import NeuralNetwork.NeuronLayer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * Created by mlx on 5/18/16.
 */
public class VLBPTrain_Map extends
        Mapper<LongWritable, Text, Text, DoubleWritable> {

    private Text Windex = new Text();
    private Text Bindex = new Text();
    private String ANN_path = "";
    private double learning_rate = 0.1;
    private double MSE_upperbound = 0.01;
    private double rho=0.7;
    private double ksi=0.04;
    private double eta=1.05;
    private double Momentum=0.65;

    protected void getANNPath(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        this.ANN_path = conf.get("ThisIterationPath");
        this.learning_rate = conf.getDouble("LearningRate", 0.1);
        this.rho=conf.getDouble("rho",0.7);
        this.ksi=conf.getDouble("ksi",0.04);
        this.eta=conf.getDouble("eta",1.05);
        this.Momentum=conf.getDouble("Momentum",0.65);
    }

    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        getANNPath(context);
        //pure text to String
        String line = value.toString();
        String[] LineDataArr = line.split(";");

        //establish a ANN from existing file
        ArtificialNeuralNetwork TrainingANN = new ArtificialNeuralNetwork(ANN_path);
        ArtificialNeuralNetwork TotalSDUpdates = new ArtificialNeuralNetwork(TrainingANN.getANN());
        ArtificialNeuralNetwork BatchStore = new ArtificialNeuralNetwork(TrainingANN.getANN());
        TotalSDUpdates.clearNetwork();

        double[][] ErrVec = new double[TrainingANN.getOutputNum()][1];
        double[][] ForwardResult = null;
        NeuronLayer[] LastUpdate = null;
        ArtificialNeuralNetwork SynthesisUpdate = null;

        double ThisIteMSE = 0.0;
        double LastIteMSE = 0.0;
        double MomentumValue=Momentum;

        double SquareError = 0.0;
        for (int EntryNum = 0; EntryNum < LineDataArr.length; EntryNum++) {
            String[] DataArr = LineDataArr[EntryNum].split("\t");
            double Tag = Double.parseDouble(DataArr[DataArr.length - 1]);
            double[][] InputVec = new double[DataArr.length - 1][1];
            for (int i = 0; i < DataArr.length - 1; i++) {
                InputVec[i][0] = Double.parseDouble(DataArr[i]);
            }
            ForwardResult = TrainingANN.getForwardResult(InputVec);
            SquareError += (Tag - ForwardResult[0][0]) * (Tag - ForwardResult[0][0]);
            TrainingANN.clearTempResult();
        }
        context.write(new Text("SquareError"), new DoubleWritable(SquareError / LineDataArr.length));
        if(SquareError / LineDataArr.length >= MSE_upperbound) {
            for (int Ite = 0; Ite < 20; Ite++) {
                double SquareErr = 0.0;
                BatchStore.clearNetwork();
                for (int EntryNum = 0; EntryNum < LineDataArr.length; EntryNum++) {
                    String[] DataArr = LineDataArr[EntryNum].split("\t");
                    double Tag = Double.parseDouble(DataArr[DataArr.length - 1]);
                    double[][] InputVec = new double[DataArr.length - 1][1];
                    for (int i = 0; i < DataArr.length - 1; i++) {
                        InputVec[i][0] = Double.parseDouble(DataArr[i]);
                    }
                    ForwardResult = TrainingANN.getForwardResult(InputVec);
                    ErrVec[0][0] = Tag - ForwardResult[0][0];

                    NeuronLayer[] ThisUpdate = TrainingANN.getSDBackwardUpdates(ErrVec, learning_rate);
                    if (EntryNum == 0 || Momentum == 0) {
                        TrainingANN.updateWeightNetwork(ThisUpdate);
                        LastUpdate = ThisUpdate;
                        BatchStore.updateWeightNetwork(ThisUpdate);
                    } else {
                        SynthesisUpdate = ArtificialNeuralNetwork.addTwoANN(ArtificialNeuralNetwork.multiplyNeuronLayers(LastUpdate, Momentum), ArtificialNeuralNetwork.multiplyNeuronLayers(ThisUpdate, 1 - Momentum));
                        TrainingANN.updateWeightNetwork(SynthesisUpdate.getANN());
                        LastUpdate = SynthesisUpdate.getANN();
                        BatchStore.updateWeightNetwork(SynthesisUpdate);
                    }
                }
                for (int i = 0; i < LineDataArr.length; i++) {
                    String[] DataArr = LineDataArr[i].split("\t");
                    double Tag = Double.parseDouble(DataArr[DataArr.length - 1]);
                    double[][] InputVec = new double[DataArr.length - 1][1];
                    for (int j = 0; j < DataArr.length - 1; j++) {
                        InputVec[j][0] = Double.parseDouble(DataArr[j]);
                    }
                    ForwardResult = TrainingANN.getForwardResult(InputVec);
                    ErrVec[0][0] = Tag - ForwardResult[0][0];
                    SquareErr += ErrVec[0][0] * ErrVec[0][0];
                    TrainingANN.clearTempResult();
                }
                ThisIteMSE = SquareErr / LineDataArr.length;
                if (Ite == 0) {
                    LastIteMSE = ThisIteMSE;
                } else {
                    double MSE_VaryRate = (ThisIteMSE - LastIteMSE) / LastIteMSE;
                    if (MSE_VaryRate > ksi) {
                        if (ThisIteMSE < 0.05) {
                            TrainingANN.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(BatchStore.getANN(), -1));
                            learning_rate *= rho;
                            Momentum = 0;
                            this.ksi=0.05;
                        } else {
                            TotalSDUpdates.updateWeightNetwork(BatchStore.getANN());
                            LastIteMSE = ThisIteMSE;
                            learning_rate *= rho;
                            Momentum = 0;
                        }
                    } else if (MSE_VaryRate < 0) {
                        TotalSDUpdates.updateWeightNetwork(BatchStore.getANN());
                        learning_rate *= eta;
                        Momentum = MomentumValue;
                        LastIteMSE = ThisIteMSE;
                    } else {
                        TotalSDUpdates.updateWeightNetwork(BatchStore.getANN());
                        Momentum = MomentumValue;
                        LastIteMSE = ThisIteMSE;
                    }
                }
            }
        }
        for (int i = 0; i < TotalSDUpdates.getANN().length; i++) {
            for (int j = 0; j < TotalSDUpdates.getANN()[i].getNeuronNum(); j++) {
                for (int k = 0; k < TotalSDUpdates.getANN()[i].getInputNum(); k++) {
                    String WeightIndex = "W-" + String.valueOf(i) + "-";
                    WeightIndex += String.valueOf(j) + "-" + String.valueOf(k);
                    Windex.set(WeightIndex);
                    context.write(Windex, new DoubleWritable(TotalSDUpdates.getANN()[i].getCertainWeight(j, k)));
                }
            }
        }

        for (int i = 0; i < TotalSDUpdates.getANN().length; i++) {
            for (int j = 0; j < TotalSDUpdates.getANN()[i].getNeuronNum(); j++) {
                String BiasIndex = "B-" + String.valueOf(i) + "-";
                BiasIndex += String.valueOf(j);
                Bindex.set(BiasIndex);
                context.write(Bindex, new DoubleWritable(TotalSDUpdates.getANN()[i].getCertainBias(j)));
            }
        }

    }
}

