import os
import pandas as pd
import pickle
from preprocessing.transformers.log_target_transformer import transform_exp



if __name__ == '__main__':

    cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main_path = "/".join(dir_path.split('/')[:-1])

    X_test = pd.read_csv("{}/data/test.csv".format(dir_path))

    model_list = ["RandomForest", "ElasticNet"]#, "Ridge", "Lasso"]
    y_pred_list = []
    for model_name in model_list:
        filename = "{}/models/finalized_{}.sav".format(dir_path, model_name)
        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        y_pred = loaded_model.predict(X_test)
        y_pred_list.append(y_pred)

    y_pred_list = pd.DataFrame(y_pred_list)
    y_pred_list = y_pred_list.transpose()

    final_y_pred = y_pred_list.mean(axis=1)


    # Submission
    submission = X_test[['Id']]
    submission.insert(1, "SalePrice", final_y_pred, True)
    submission = transform_exp(submission, 'SalePrice')
    submission.to_csv("{}/data/submission.csv".format(dir_path), index=False)

    print(submission)
