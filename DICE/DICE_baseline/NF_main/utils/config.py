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
class diabetes:
    """
    Configuration of dataset diabetes
    """

    # the size of total features
    params = 8

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([1, 9])
    input_bounds.append([1, 9]) 
    input_bounds.append([1, 9])
    input_bounds.append([1, 9])
    input_bounds.append([1, 9])
    input_bounds.append([1, 9])
    input_bounds.append([1, 9])

    # the name of each feature
    feature_name = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']

    # the name of each class
    class_name = ["not disease", "disease"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7]

    
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

class students:
    """
    Configuration of dataset student performance
    """

    # the size of total features
    params = 32

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 1])#
    input_bounds.append([0, 1])#
    input_bounds.append([15, 22])#
    input_bounds.append([0, 1])#
    input_bounds.append([0, 1])#
    input_bounds.append([0, 1])#
    input_bounds.append([0, 4])#
    input_bounds.append([0, 4])#
    input_bounds.append([1, 5])#
    input_bounds.append([1, 5])#
    input_bounds.append([1, 4])#
    input_bounds.append([1, 3])#
    input_bounds.append([1, 4])#
    input_bounds.append([1, 4])#
    input_bounds.append([1, 4])#
    input_bounds.append([0, 1])#
    input_bounds.append([0, 1])#
    input_bounds.append([0, 1])#
    input_bounds.append([0, 1])#
    input_bounds.append([0, 1])#
    input_bounds.append([0, 1])#
    input_bounds.append([0, 1])#
    input_bounds.append([0, 1])#
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([0, 93])
    input_bounds.append([0, 20])
    input_bounds.append([0, 20])
    


    # the name of each feature
    feature_name = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                   'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
                   'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                   'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
                   'Walc', 'health', 'absences', 'G1', 'G2']

    # the name of each class
    class_name = ["not pass", "pass"]
    
    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
                            17, 18, 19, 20, 21, 22 ,23, 24, 25, 26, 27, 28, 29, 30, 31]
class meps15:
    """
    Configuration of dataset compas
    """

    # the size of total features
    params = 137

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 1])
    input_bounds.append([1, 10])
    input_bounds.append([1, 10])
    input_bounds.append([0, 26])
    for feat in range(132):
        input_bounds.append([0, 1])


    # the name of each feature
    feature_name = ['AGE', 'RACE', 'PCS42', 'MCS42', 'K6SUM42', 'REGION=1', 'REGION=2',
       'REGION=3', 'REGION=4', 'SEX=1', 'MARRY=1', 'MARRY=2', 'MARRY=3',
       'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7', 'MARRY=8', 'MARRY=9',
       'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2', 'FTSTU=3',
       'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4', 'HONRDC=1',
       'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1', 'RTHLTH=1',
       'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5', 'MNHLTH=-1',
       'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4', 'MNHLTH=5',
       'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1', 'CHDDX=1',
       'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2', 'MIDX=-1',
       'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
       'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
       'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
       'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
       'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
       'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
       'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
       'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
       'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
       'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
       'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
       'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
       'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
       'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
       'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
       'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
       'INSCOV=1', 'INSCOV=2', 'INSCOV=3']

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,       27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,     58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,     89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,     116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136]
    
    
    
class meps16:
    """
    Configuration of dataset compas
    """

    # the size of total features
    params = 137

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 1])
    input_bounds.append([1, 10])
    input_bounds.append([1, 10])
    input_bounds.append([0, 26])
    for feat in range(132):
        input_bounds.append([0, 1])


    # the name of each feature
    feature_name = ['AGE', 'RACE', 'PCS42', 'MCS42', 'K6SUM42', 'REGION=1', 'REGION=2',
       'REGION=3', 'REGION=4', 'SEX=1', 'MARRY=1', 'MARRY=2', 'MARRY=3',
       'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7', 'MARRY=8', 'MARRY=9',
       'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2', 'FTSTU=3',
       'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4', 'HONRDC=1',
       'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1', 'RTHLTH=1',
       'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5', 'MNHLTH=-1',
       'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4', 'MNHLTH=5',
       'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1', 'CHDDX=1',
       'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2', 'MIDX=-1',
       'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
       'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
       'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
       'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
       'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
       'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
       'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
       'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
       'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
       'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
       'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
       'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
       'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
       'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
       'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
       'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
       'INSCOV=1', 'INSCOV=2', 'INSCOV=3']

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,       27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,     58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,     89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,     116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136]
    
    