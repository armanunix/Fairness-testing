class census:
    """
    Configuration of dataset Census Income
    """

    # the size of total features
    params = 13

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 7])
    input_bounds.append([0, 39]) #69 for THEMIS
    input_bounds.append([0, 15])
    input_bounds.append([0, 6])
    input_bounds.append([0, 13])
    input_bounds.append([0, 5])
    input_bounds.append([0, 4])
    input_bounds.append([0, 1])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])

    # the name of each feature
    feature_name = ["age", "workclass", "fnlwgt", "education", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain",
                                                                      "capital_loss", "hours_per_week", "native_country"]

    # the name of each class
    class_name = ["low", "high"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

class credit:
    """
    Configuration of dataset German Credit
    """

    # the size of total features
    params = 20

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 3])
    input_bounds.append([1, 80])
    input_bounds.append([0, 4])
    input_bounds.append([0, 10])
    input_bounds.append([1, 200])
    input_bounds.append([0, 4])
    input_bounds.append([0, 4])
    input_bounds.append([1, 4])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 8])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 2])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])

    # the name of each feature
    feature_name = ["checking_status", "duration", "credit_history", "purpose", "credit_amount", "savings_status", "employment", "installment_commitment", "sex", "other_parties",
                                                                      "residence", "property_magnitude", "age", "other_payment_plans", "housing", "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker"]

    # the name of each class
    class_name = ["bad", "good"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19]

class bank:
    """
    Configuration of dataset Bank Marketing
    """

    # the size of total features
    params = 16

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 11])
    input_bounds.append([0, 2])
    input_bounds.append([0, 3])
    input_bounds.append([0, 1])
    input_bounds.append([-20, 179])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 31])
    input_bounds.append([0, 11])
    input_bounds.append([0, 99])
    input_bounds.append([1, 63])
    input_bounds.append([-1, 39])
    input_bounds.append([0, 1])
    input_bounds.append([0, 3])

    # the name of each feature
    feature_name = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
                                                                      "month", "duration", "campaign", "pdays", "previous", "poutcome"]

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
class compas:
    """
    Configuration of dataset compas
    """

    # the size of total features
    params = 12

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([0, 1])
    input_bounds.append([0, 20])
    input_bounds.append([1, 10])
    input_bounds.append([0, 38])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([1, 10])
    input_bounds.append([1, 10])
    input_bounds.append([0, 38])

    # the name of each feature
    feature_name = ["sex", "age", "race", "att4", "att5", "att6", "att7", "att8", "att9", "att10",
                                                                      "att11", "att12"]

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
class default:
    """
    Configuration of dataset default credit
    """

    # the size of total features
    params = 23

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 100])
    input_bounds.append([0, 1])
    input_bounds.append([0, 6])
    input_bounds.append([0, 3])
    input_bounds.append([2, 7])
    input_bounds.append([0, 10])
    input_bounds.append([0, 10])
    input_bounds.append([0, 10])
    input_bounds.append([0, 10])
    input_bounds.append([0, 10])
    input_bounds.append([0, 10])
    input_bounds.append([-16, 96])
    input_bounds.append([-6, 98])
    input_bounds.append([-15, 166])
    input_bounds.append([-17, 89])
    input_bounds.append([-8, 92])
    input_bounds.append([-33, 96])
    input_bounds.append([0, 87])
    input_bounds.append([0, 168])
    input_bounds.append([0, 89])
    input_bounds.append([0, 62])
    input_bounds.append([0, 42])
    input_bounds.append([0, 52])


    # the name of each feature
    feature_name = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5",
                      "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1",
                       "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

class heart:
    """
    Configuration of dataset Heart Disease
    """

    # the size of total features
    params = 13

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 7])
    input_bounds.append([0, 1])
    input_bounds.append([1, 4]) 
    input_bounds.append([94, 200])
    input_bounds.append([126, 564])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([71, 202])
    input_bounds.append([0, 1])
    input_bounds.append([0, 6])
    input_bounds.append([1, 3])
    input_bounds.append([0, 3])
    input_bounds.append([3, 7])

    # the name of each feature
    feature_name = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
                                                                      "slope", "ca", "thal"]

    # the name of each class
    class_name = ["disease", "not disease"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]