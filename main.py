from ANNPh3 import N_Network
import numpy as np
import time

def main():
    start_time= time.time()
    # X = (hours studying, hours sleeping up seven days before the exam),
    # y = score on test
    # xPredicted = seven days hours of studying & seven days  hours of sleeping (input data for prediction)
    X = np.array(([2,9,0,6,1,5,1,5,2,5,3,8,2,5],
                    [1,7,3,6,2,6,4,7,2,8,3,9,1,5],
                    [2,6,3,7,1,0,6,2,4,6,2,5,3,9],
                    [3,5,1,8,2,4,2,8,4,7,2,8,2,9],
                    [2,8,2,7,1,5,1,7,4,8,2,7,1,5],
                    [1,9,1,5,3,7,3,9,4,9,2,8,3,6],
                    [1,7,4,7,4,8,1,8,4,5,2,9,2,6],
                    [4,5,2,9,1,4,3,6,4,8,2,5,1,8],
                    [3,6,3,6,3,4,2,7,4,5,2,5,3,7]   ),
                    dtype=float)
    y = np.array(([71], [92], [80], [87], [80], [90], [77], [88], [91]), dtype=float)
    xPrediction = np.array(([1,5,3,6,2,7,0,9,3,6,1,8,2,7]), dtype=float)

    # scale units
    X = X/np.amax(X, axis=0) # maximum number of X array
    xPrediction = xPrediction/np.amax(xPrediction, axis=0) # maximum of xPredicted (our input data for the prediction)
    y = y/100 # max test score is 100

    ANN= N_Network()
    for i in range(1000): # This trians the ANN 10,000
      print(" #" + str(i) + "\n")
      print("Input (normalized): \n" + str(X))
      print ("Actual Outputs: \n" + str(y))
      print("Predicted Outputs: \n" + str(ANN.forward_propagation(X)))
      print("total error: \n" + str(np.mean(np.square(y - ANN.forward_propagation(X)))))
      print ("\n")
      ANN.train(X, y)

    ANN.save_updated_Weights()
    ANN.make_prediction(xPrediction)
    print("--- %s seconds ---" % (time.time() - start_time))


main()
