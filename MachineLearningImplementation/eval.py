from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ML.preprocessing import ProcessingUnit
from ML.evaluation import EvaluationUnit
import pandas as pd

if __name__ == '__main__':
    t = pd.read_csv('./testfile/dataset.csv')
    t = t.to_numpy(dtype=float)
    # t = np.loadtxt('./testfile/dataset.csv', delimiter=',',encoding='utf-8')
    pu = ProcessingUnit(t)
    x = pu.x
    y = pu.y
    # pu.y_to_onehot()
    # pu.split_dataset(ratio=(0.8, 0.1, 0.1))
    # pu.z_score()
    # xtrain, ytrain = pu.train
    # xcv, ycv = pu.cv
    # xtest, ytest = pu.test
    # m_train = xtrain.shape[0]
    # xt, xtcv = np.hsplit([2000])
    # yt, ytcv = np.hsplit([2000])
    std = StandardScaler()
    std.fit(x)
    sx = std.transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y.flatten(), test_size=0.1,
                                                    stratify=y.flatten())
    nn = MLPClassifier(solver='adam', learning_rate_init=0.001, learning_rate='constant', alpha=0,
                       hidden_layer_sizes=(72,), random_state=1,
                       max_iter=500, batch_size='auto', verbose=True)
    nn.fit(xtrain, ytrain)

    y_p_train = nn.predict(xtrain)
    # y_p_cv = nn.predict(xcv)
    y_p_test = nn.predict(xtest)

    eu1 = EvaluationUnit(y_p_train.T, ytrain)
    # eu2 = EvaluationUnit(y_p_cv.T, ycv)
    eu3 = EvaluationUnit(y_p_test.T, ytest)
    # p, r = eu3.p_and_r()

    # print(f'acc_train -> {eu1.accuracy()}')
    # print(f'acc_cv -> {eu2.accuracy()}')
    # print(f'acc_test -> {eu3.accuracy()}')
    # print(f'confusing_matrix -> "\n"{eu3.confusing_matrix()}')
    # print('precision -> ' + str(list(p)))
    # print('recall -> ' + str(list(r)))
    # print(accuracy_score(ytest, y_p_test))
    # print(accuracy_score(ytrain, y_p_train))
    # print(accuracy_score(ycv, y_p_cv))
    # print(accuracy_score(ytest, y_p_test))
    # print(precision_score(ytest, y_p_test, average='samples'))
    # print(f'mse -> {eu1.mean_square_error()}')
    # print(f'mse -> {eu2.mean_square_error()}')
    # print(f'mse -> {eu3.mean_square_error()}')
    # print(mean_squared_error(y_true=ytest, y_pred=y_p_test))
