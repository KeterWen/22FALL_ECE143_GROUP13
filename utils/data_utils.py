from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

def generate_dataset(df_maths, df_port, random_seed=0):
    """
    generate train/test dataset
    :param df_maths: DataFrame of maths
    :param df_port: DataFrame of portuguese
    :param random_seed: random seed when split datasets
    :return:
    """

    df_maths_label = df_maths.copy()
    df_port_label = df_port.copy()

    LabelEnc = preprocessing.LabelEncoder()

    categorical_features = df_maths_label.select_dtypes(include=['object']).columns
    for Feature in categorical_features:
        df_maths_label[Feature] = LabelEnc.fit_transform(df_maths_label[Feature])

    for Feature in categorical_features:
        df_port_label[Feature] = LabelEnc.fit_transform(df_port_label[Feature])

    ss = preprocessing.StandardScaler()

    Xm = df_maths_label.copy()
    Xm.drop(['G1', 'G2', 'G3'],axis = 1,inplace = True)
    ym = df_maths_label['G3']
    X_trainm, X_testm, y_trainm, y_testm = train_test_split(Xm, ym, test_size=0.2,random_state=random_seed)
    X_trainm = ss.fit_transform(X_trainm)
    X_testm = ss.transform(X_testm)

    Xp = df_port_label.copy()
    Xp.drop(['G1', 'G2', 'G3'],axis = 1,inplace = True)
    yp = df_port_label['G3']

    X_trainp, X_testp, y_trainp, y_testp = train_test_split(Xp, yp, test_size=0.2,random_state=random_seed)
    X_trainp = ss.fit_transform(X_trainp)
    X_testp = ss.transform(X_testp)

    return X_trainm, X_testm, X_trainp, X_testp

def read_csv(filename):
    '''
    collect the csv file data
    :param filename: name of csv file 
    :return: dataframe of csv file
    '''
    subject = pd.read_csv(filename, index_col=None)
    return subject