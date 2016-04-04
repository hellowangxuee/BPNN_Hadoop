package NeuralNetwork;

import java.util.Map;
import java.util.regex.Matcher;

/**
 * Created by mlx on 3/30/16.
 */
public class LinearSearchMinumum {
    public static double[] getIntervalLocation(ArtificialNeuralNetwork InitalANN,double[][] InputVec,double Tag,NeuronLayer[] SearchDirection,double HeuristicStep) {
        double[] IntervalLoation = new double[2];
        IntervalLoation[0] = 0.0;
        IntervalLoation[1] = HeuristicStep;

        double LastStartPoint = 0.0;

        ArtificialNeuralNetwork testANN = new ArtificialNeuralNetwork(InitalANN.getANN());
        double[][] ErrVec_a = new double[InitalANN.getOutputNum()][1];
        double[][] ErrVec_b = new double[InitalANN.getOutputNum()][1];

        double[][] ForwardResult = InitalANN.getForwardResult(InputVec);
        ErrVec_a[0][0] = Tag - ForwardResult[0][0];

        testANN.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, IntervalLoation[1]));
        ForwardResult = testANN.getForwardResult(InputVec);
        ErrVec_b[0][0] = Tag - ForwardResult[0][0];

        for (int i = 1; Math.abs(ErrVec_a[0][0]) >= Math.abs(ErrVec_b[0][0]); i++) {
            ErrVec_a[0][0] = ErrVec_b[0][0];
            LastStartPoint = IntervalLoation[0];
            IntervalLoation[0] = IntervalLoation[1];
            IntervalLoation[1] = Math.pow(2, i) * HeuristicStep;

            testANN.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, IntervalLoation[1] - IntervalLoation[0]));
            ForwardResult = testANN.getForwardResult(InputVec);
            ErrVec_b[0][0] = Tag - ForwardResult[0][0];
        }
        IntervalLoation[0]=LastStartPoint;
        return IntervalLoation;
    }

    public static double getTolerableMinimum(ArtificialNeuralNetwork InitalANN,double[][] InputVec,double Tag,NeuronLayer[] SearchDirection,double[] IntervalLocation,double Tol){
        double tau=0.618;
        double left_point=IntervalLocation[0];
        double right_point=IntervalLocation[1];
        double bet_point_left=left_point+(1-tau)*(right_point-left_point);
        double bet_point_right=right_point-(1-tau)*(right_point-left_point);

        double last_bet_point_left=0.0;
        double last_bet_point_right=0.0;

        ArtificialNeuralNetwork testANN_left = new ArtificialNeuralNetwork(InitalANN.getANN());
        ArtificialNeuralNetwork testANN_right = new ArtificialNeuralNetwork(InitalANN.getANN());

        double[][] ErrVec_c = new double[InitalANN.getOutputNum()][1];
        double[][] ErrVec_d = new double[InitalANN.getOutputNum()][1];

        testANN_left.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, bet_point_left - last_bet_point_left));
        testANN_right.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, bet_point_right - last_bet_point_right));

        ErrVec_c[0][0] = Tag - testANN_left.getForwardResult(InputVec)[0][0];
        ErrVec_d[0][0] = Tag - testANN_right.getForwardResult(InputVec)[0][0];

        while (right_point-left_point>Tol) {
            last_bet_point_left=bet_point_left;
            last_bet_point_right=bet_point_right;
            if(Math.abs(ErrVec_c[0][0]) < Math.abs(ErrVec_d[0][0])){
                right_point=bet_point_right;
                bet_point_right=bet_point_left;
                bet_point_left=left_point+(1-tau)*(right_point-left_point);

                ErrVec_d[0][0]=ErrVec_c[0][0];

                testANN_left.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection,(bet_point_left-last_bet_point_left)));
                ErrVec_c[0][0] = Tag - testANN_left.getForwardResult(InputVec)[0][0];
            }
            else{
                left_point=bet_point_left;
                bet_point_left=bet_point_right;
                bet_point_right=right_point-(1-tau)*(right_point-left_point);

                ErrVec_c[0][0]=ErrVec_d[0][0];

                testANN_right.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(SearchDirection, bet_point_right - last_bet_point_right));
                ErrVec_d[0][0] = Tag - testANN_right.getForwardResult(InputVec)[0][0];
            }
        }
        return ((right_point+left_point)/2);

    }
}
