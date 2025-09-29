# Build Machine Learning Algorithms With Numpy
## File Structure
```
/
│  .gitignore
│  README.md
│      
├─resources
│  └─testing-data                          # Testing data
│          dataset.csv
│          heart.csv
│          house.csv
│          x.csv
│          y.csv
│          
└─src
    ├─models                               # Model implementations
    │      Kmeans.py
    │      LinearRegression.py
    │      LogisticRegression.py
    │      NeuralNetwork.py
    │      Optimizer.py                    # Tncludes sgd, momentum, rmsprop, and adam
    │      __init__.py
    │      
    ├─test
    │      test.py
    │      
    └─utils                                 # Helper functions and classes
            BaseFunctions.py                # Logistic, tanh, linear, relu, and softmax
            Evaluation.py                   # Calculate mse, acc, confusing matrix, precision, and recall
            ParamInitializer.py             # Intialize model parameters using xavier_init, he_init, or random_init
            Preprocessing.py                # Preprocess data to match model's input
            __init__.py
```
## About This Project
I started this project when I was studying machine learning for the first time; with an aim of gaining deep understanding of the maths and structures under the hood. It's been a while since I last touch on the codes. But when I browsed what was written, I can still get a fresh delivery of understanding. So I guess this method of studying is of great benefit, and I will keep updating in the future for myself.