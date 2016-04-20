package NeuralNetwork;

import java.util.Map;
import java.util.Vector;
import java.util.regex.Matcher;

/**
 * Created by mlx on 3/30/16.
 */
public class LinearSearchMinumum {

    public static double calculateMSEOverCertainDataset(ArtificialNeuralNetwork ANN,Vector<Double[]> Dataset) {
        double TSE = 0;
        for (int i = 0; i < Dataset.size(); i++) {
            double[][] InputVec = new double[ANN.getInputNum()][1];
            for (int j = 0; j < ANN.getInputNum(); j++) {
                InputVec[j][0] = ((Dataset.get(i)))[j];
            }
            double[][] ForwardResult = ANN.getForwardResult(InputVec);
            TSE += Math.pow((Dataset.get(i))[ANN.getInputNum()] - ForwardResult[0][0], 2);
        }
        return TSE / Dataset.size();
    }

    public static double[] getIntervalLocation(ArtificialNeuralNetwork InitalANN, Vector<Double[]> InputDataVec, NeuronLayer[] SearchDirection, double HeuristicStep) {
        double[] IntervalLoation = new double[2];
        IntervalLoation[0] = 0.0;
        IntervalLoation[1] = HeuristicStep;

        double LastStartPoint = 0.0;

        ArtificialNeuralNetwork testANN = new ArtificialNeuralNetwork(InitalANN.getANN());
        testANN.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, IntervalLoation[1]));

        double[][] ErrVec_a = new double[InitalANN.getOutputNum()][1];
        double[][] ErrVec_b = new double[InitalANN.getOutputNum()][1];
        double ErrSum_a=0;
        double ErrSum_b=0;

        for(int i=0;i<InputDataVec.size();i++) {
            double[][] InputVec=new double[InitalANN.getInputNum()][1];
            for(int j=0;j<InitalANN.getInputNum();j++) {
                InputVec[j][0] = ((Double[]) (InputDataVec.get(i)))[j];
            }
            double[][] ForwardResult1 = InitalANN.getForwardResult(InputVec);
            double[][] ForwardResult2 = testANN.getForwardResult(InputVec);

            ErrSum_a += Math.pow(((Double[]) (InputDataVec.get(i)))[InitalANN.getInputNum()] - ForwardResult1[0][0],2) ;
            ErrSum_b += Math.pow(((Double[]) (InputDataVec.get(i)))[InitalANN.getInputNum()] - ForwardResult2[0][0],2) ;
        }

        for (int i = 1; ErrSum_a >= ErrSum_b; i++) {
            ErrSum_a = ErrSum_b;
            LastStartPoint = IntervalLoation[0];
            IntervalLoation[0] = IntervalLoation[1];
            IntervalLoation[1] = Math.pow(2, i) * HeuristicStep;

            testANN.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, IntervalLoation[1] - IntervalLoation[0]));

            ErrSum_b=0;
            for(int k=0;k<InputDataVec.size();k++) {
                double[][] InputVec=new double[InitalANN.getInputNum()][1];
                for(int j=0;j<InitalANN.getInputNum();j++) {
                    InputVec[j][0] = ((Double[]) (InputDataVec.get(k)))[j];
                }
                double[][] ForwardResult2 = testANN.getForwardResult(InputVec);
                ErrSum_b += Math.pow(((Double[]) (InputDataVec.get(k)))[InitalANN.getInputNum()] - ForwardResult2[0][0],2) ;
            }
        }
        IntervalLoation[0]=LastStartPoint;
        return IntervalLoation;
    }

    public static double getTolerableMinimum(ArtificialNeuralNetwork InitalANN,Vector<Double[]> InputDataVec,NeuronLayer[] SearchDirection,double[] IntervalLocation,double Tol) {
        double tau = 0.618;
        double left_point = IntervalLocation[0];
        double right_point = IntervalLocation[1];
        double bet_point_left = left_point + (1 - tau) * (right_point - left_point);
        double bet_point_right = right_point - (1 - tau) * (right_point - left_point);

        double last_bet_point_left = 0.0;
        double last_bet_point_right = 0.0;

        ArtificialNeuralNetwork testANN_left = new ArtificialNeuralNetwork(InitalANN.getANN());
        ArtificialNeuralNetwork testANN_right = new ArtificialNeuralNetwork(InitalANN.getANN());

//        double[][] ErrVec_c = new double[InitalANN.getOutputNum()][1];
//        double[][] ErrVec_d = new double[InitalANN.getOutputNum()][1];

        testANN_left.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, bet_point_left - last_bet_point_left));
        testANN_right.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, bet_point_right - last_bet_point_right));

        double ErrSum_c = 0;
        double ErrSum_d = 0;
        for (int i = 0; i < InputDataVec.size(); i++) {
            double[][] InputVec = new double[InitalANN.getInputNum()][1];
            for (int j = 0; j < InitalANN.getInputNum(); j++) {
                InputVec[j][0] = ((Double[]) (InputDataVec.get(i)))[j];
            }
            double[][] ForwardResult1 = testANN_left.getForwardResult(InputVec);
            double[][] ForwardResult2 = testANN_right.getForwardResult(InputVec);

            ErrSum_c += Math.pow(((Double[]) (InputDataVec.get(i)))[InitalANN.getInputNum()] - ForwardResult1[0][0], 2);
            ErrSum_d += Math.pow(((Double[]) (InputDataVec.get(i)))[InitalANN.getInputNum()] - ForwardResult2[0][0], 2);
        }
//
//        ErrVec_c[0][0] = Tag - testANN_left.getForwardResult(InputVec)[0][0];
//        ErrVec_d[0][0] = Tag - testANN_right.getForwardResult(InputVec)[0][0];

        while (right_point - left_point > Tol) {
            last_bet_point_left = bet_point_left;
            last_bet_point_right = bet_point_right;
            if (ErrSum_c < ErrSum_d) {
                right_point = bet_point_right;
                bet_point_right = bet_point_left;
                bet_point_left = left_point + (1 - tau) * (right_point - left_point);

                ErrSum_d = ErrSum_c;

                testANN_left.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, (bet_point_left - last_bet_point_left)));
                ErrSum_c = 0;
                for (int i = 0; i < InputDataVec.size(); i++) {
                    double[][] InputVec = new double[InitalANN.getInputNum()][1];
                    for (int j = 0; j < InitalANN.getInputNum(); j++) {
                        InputVec[j][0] = ((Double[]) (InputDataVec.get(i)))[j];
                    }
                    double[][] ForwardResult1 = testANN_left.getForwardResult(InputVec);
                    ErrSum_c += Math.pow(((Double[]) (InputDataVec.get(i)))[InitalANN.getInputNum()] - ForwardResult1[0][0], 2);
                }
            } else {
                left_point = bet_point_left;
                bet_point_left = bet_point_right;
                bet_point_right = right_point - (1 - tau) * (right_point - left_point);

                ErrSum_c = ErrSum_d;

                testANN_right.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, bet_point_right - last_bet_point_right));
                ErrSum_d = 0;
                for (int i = 0; i < InputDataVec.size(); i++) {
                    double[][] InputVec = new double[InitalANN.getInputNum()][1];
                    for (int j = 0; j < InitalANN.getInputNum(); j++) {
                        InputVec[j][0] = ((Double[]) (InputDataVec.get(i)))[j];
                    }
                    double[][] ForwardResult2 = testANN_right.getForwardResult(InputVec);
                    ErrSum_d += Math.pow(((Double[]) (InputDataVec.get(i)))[InitalANN.getInputNum()] - ForwardResult2[0][0], 2);
                }
            }
        }
        return ((right_point + left_point) / 2);

    }
}
