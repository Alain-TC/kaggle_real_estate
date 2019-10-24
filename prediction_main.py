import os
import pandas as pd
import pickle
from preprocessing.transformers.log_target_transformer import transform_exp



if __name__ == '__main__':

    cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main_path = "/".join(dir_path.split('/')[:-1])

    X_test = pd.read_csv("{}/data/test.csv".format(dir_path))


    model_name = "RandomForest"

    filename = "{}/models/finalized_{}.sav".format(dir_path, model_name)
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict(X_test)


    # Submission
    submission = X_test[['Id']]
    submission.insert(1, "SalePrice", y_pred, True)
    submission = transform_exp(submission, 'SalePrice')
    submission.to_csv("{}/data/submission.csv".format(dir_path), index=False)

    print(submission)
