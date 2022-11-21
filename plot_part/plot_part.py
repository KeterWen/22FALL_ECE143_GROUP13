import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import os
import warnings

from IPython.display import Image, display
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from matplotlib.pyplot import MultipleLocator
warnings.filterwarnings('ignore')


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

def plot_hist(df_maths, df_port, column_i,  save_path='./plot/test.png'):
    """
    plot and save the hist given column_i
    :param df_maths: DataFrame of maths
    :param df_port: DataFrame of portuguese
    :param column_i: data to plot
    :param save_path: save path
    :return:
    """
    plt.title(column_i)
    plt.hist([df_maths[column_i], df_port[column_i]], label=['Maths', 'Port'])
    # plt.legend(labels=['Maths', 'Port'], loc=2, bbox_to_anchor=(1, 1))
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_heatmap(df, save_path='./plot/heatmap.png'):
    """
    plot and save the heatmap given useful columns
    :param df_maths: DataFrame to plot
    :param save_path: save path
    :return:
    """
    useful_columns = ['age', 'famsize', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
    df_corr = df[useful_columns]
    plt.figure(figsize=(15,15))

    sns.heatmap(df_corr.corr(),annot=True,cmap='GnBu')

    plt.savefig(save_path)
    plt.close()

def plot_bar_goout_vs_DWalc(df_maths, df_port, save_path='./plot/bar_goout_vs_DWalc.png'):
    """
    plot and save figure the bar chart between goout and Dalc / Walc
    :param df_maths: DataFrame of maths
    :param df_port: DataFrame of portuguese
    :param save_path: save path
    :return:
    """
    goout_DWalc_dict = {}
    goout_list = list(range(1, 6))
    keys_list = ['avgDalc','avgWalc']
    for goout_i in range(1, 6):
      goout_DWalc_dict[goout_i] = {}
      Dalc_list = list(df_maths[df_maths['goout']==goout_i]['Dalc']) + list(df_port[df_port['goout']==goout_i]['Dalc'])
      Walc_list = list(df_maths[df_maths['goout']==goout_i]['Walc']) + list(df_port[df_port['goout']==goout_i]['Walc'])
      goout_DWalc_dict[goout_i]['avgDalc'] = np.round(np.mean(Dalc_list), 3)
      goout_DWalc_dict[goout_i]['avgWalc'] = np.round(np.mean(Walc_list), 3)
    
    barX = np.arange(1, 6)
    width = 0
    wid = 0.3
    
    for i in keys_list:
      plt.bar(barX+width , [goout_DWalc_dict[j][i]  for j in goout_list], width = 0.3, label = f'{i}')
      width += wid
    plt.xlabel('goout')
    plt.ylabel('avg of DWalc')
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_bar_studytime_vs_G(df_maths, df_port, save_path='./plot/bar_studytime_vs_G.png'):
    """
    plot and save figure the bar chart between studytime and G123
    :param df_maths: DataFrame of maths
    :param df_port: DataFrame of portuguese
    :param save_path: save path
    :return:
    """
    studytime_G123_dict = {}
    studytime_list = list(range(1, 5))
    keys_list = ['avgMathsG1','avgMathsG2', 'avgMathsG3', 'avgPortG1', 'avgPortG2', 'avgPortG3']
    for studytime_i in range(1, 5):
      studytime_G123_dict[studytime_i] = {}
      studytime_G123_dict[studytime_i]['avgMathsG1'] = np.round(np.mean(df_maths[df_maths['studytime']==studytime_i]['G1']), 3)
      studytime_G123_dict[studytime_i]['avgMathsG2'] = np.round(np.mean(df_maths[df_maths['studytime']==studytime_i]['G2']), 3)
      studytime_G123_dict[studytime_i]['avgMathsG3'] = np.round(np.mean(df_maths[df_maths['studytime']==studytime_i]['G3']), 3)
      studytime_G123_dict[studytime_i]['avgPortG1'] = np.round(np.mean(df_port[df_port['studytime']==studytime_i]['G1']), 3)
      studytime_G123_dict[studytime_i]['avgPortG2'] = np.round(np.mean(df_port[df_port['studytime']==studytime_i]['G2']), 3)
      studytime_G123_dict[studytime_i]['avgPortG3'] = np.round(np.mean(df_port[df_port['studytime']==studytime_i]['G3']), 3)

    barX = np.arange(1, 5)
    width = 0
    wid = 0.1

    color_list = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for i in range(len(keys_list)):
      key_i = keys_list[i]
      if key_i == 'avgMathsG3':
          lr = LinearRegression()
          lr.fit(np.array(df_maths['studytime']).reshape(-1, 1), np.array(df_maths['G3']))
          plt.plot(barX + width, lr.coef_ * barX + lr.intercept_, color_list[i])
      elif key_i == 'avgPortG3':
          lr = LinearRegression()
          lr.fit(np.array(df_port['studytime']).reshape(-1, 1), np.array(df_port['G3']))
          plt.plot(barX + width, lr.coef_ * barX + lr.intercept_, color_list[i])

      plt.bar(barX + width, [studytime_G123_dict[j][key_i] for j in studytime_list], width=0.1, label=key_i, color=color_list[i])
      width += wid
    plt.xlabel('studytime')
    plt.ylabel('avg of G')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.ylim(0, 22)
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_bar_Dalc_vs_G(df_maths, df_port, save_path='./plot/bar_Dalc_vs_G.png'):
    """
    plot and save figure the bar chart between Dalc and G123
    :param df_maths: DataFrame of maths
    :param df_port: DataFrame of portuguese
    :param save_path: save path
    :return:
    """
    Dalc_G123_dict = {}
    Dalc_list = list(range(1, 6))
    keys_list = ['avgMathsG1','avgMathsG2', 'avgMathsG3', 'avgPortG1', 'avgPortG2', 'avgPortG3']
    for Dalc_i in range(1, 6):
      Dalc_G123_dict[Dalc_i] = {}
      Dalc_G123_dict[Dalc_i]['avgMathsG1'] = np.round(np.mean(df_maths[df_maths['Dalc']==Dalc_i]['G1']), 3)
      Dalc_G123_dict[Dalc_i]['avgMathsG2'] = np.round(np.mean(df_maths[df_maths['Dalc']==Dalc_i]['G2']), 3)
      Dalc_G123_dict[Dalc_i]['avgMathsG3'] = np.round(np.mean(df_maths[df_maths['Dalc']==Dalc_i]['G3']), 3)
      Dalc_G123_dict[Dalc_i]['avgPortG1'] = np.round(np.mean(df_port[df_port['Dalc']==Dalc_i]['G1']), 3)
      Dalc_G123_dict[Dalc_i]['avgPortG2'] = np.round(np.mean(df_port[df_port['Dalc']==Dalc_i]['G2']), 3)
      Dalc_G123_dict[Dalc_i]['avgPortG3'] = np.round(np.mean(df_port[df_port['Dalc']==Dalc_i]['G3']), 3)

    barX = np.arange(1, 6)
    width = 0
    wid = 0.1

    color_list = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for i in range(len(keys_list)):
        key_i = keys_list[i]
        if key_i == 'avgMathsG3':
            lr = LinearRegression()
            lr.fit(np.array(df_maths['Dalc']).reshape(-1, 1), np.array(df_maths['G3']))
            plt.plot(barX + width, lr.coef_ * barX + lr.intercept_, color_list[i])
        elif key_i == 'avgPortG3':
            lr = LinearRegression()
            lr.fit(np.array(df_port['Dalc']).reshape(-1, 1), np.array(df_port['G3']))
            plt.plot(barX + width, lr.coef_ * barX + lr.intercept_, color_list[i])
        plt.bar(barX + width, [Dalc_G123_dict[j][key_i] for j in Dalc_list], width=0.1, label=key_i,
                color=color_list[i])
        width += wid

    plt.xlabel('Dalc')
    plt.ylabel('avg of G')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.ylim(0, 20)
    plt.legend(fontsize='small')
    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_bar_Walc_vs_G(df_maths, df_port, save_path='./plot/bar_Walc_vs_G.png'):
    """
    plot and save figure the bar chart between Walc and G123
    :param df_maths: DataFrame of maths
    :param df_port: DataFrame of portuguese
    :param save_path: save path
    :return:
    """
    Walc_G123_dict = {}
    Walc_list = list(range(1, 6))
    keys_list = ['avgMathsG1','avgMathsG2', 'avgMathsG3', 'avgPortG1', 'avgPortG2', 'avgPortG3']
    for Walc_i in range(1, 6):
      Walc_G123_dict[Walc_i] = {}
      Walc_G123_dict[Walc_i]['avgMathsG1'] = np.round(np.mean(df_maths[df_maths['Walc']==Walc_i]['G1']), 3)
      Walc_G123_dict[Walc_i]['avgMathsG2'] = np.round(np.mean(df_maths[df_maths['Walc']==Walc_i]['G2']), 3)
      Walc_G123_dict[Walc_i]['avgMathsG3'] = np.round(np.mean(df_maths[df_maths['Walc']==Walc_i]['G3']), 3)
      Walc_G123_dict[Walc_i]['avgPortG1'] = np.round(np.mean(df_port[df_port['Walc']==Walc_i]['G1']), 3)
      Walc_G123_dict[Walc_i]['avgPortG2'] = np.round(np.mean(df_port[df_port['Walc']==Walc_i]['G2']), 3)
      Walc_G123_dict[Walc_i]['avgPortG3'] = np.round(np.mean(df_port[df_port['Walc']==Walc_i]['G3']), 3)

    barX = np.arange(1, 6)
    width = 0
    wid = 0.1

    color_list = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for i in range(len(keys_list)):
        key_i = keys_list[i]

        if key_i == 'avgMathsG3':
            lr = LinearRegression()
            lr.fit(np.array(df_maths['Walc']).reshape(-1, 1), np.array(df_maths['G3']))
            plt.plot(barX + width, lr.coef_ * barX + lr.intercept_, color_list[i])
        elif key_i == 'avgPortG3':
            lr = LinearRegression()
            lr.fit(np.array(df_port['Walc']).reshape(-1, 1), np.array(df_port['G3']))
            plt.plot(barX + width, lr.coef_ * barX + lr.intercept_, color_list[i])

        plt.bar(barX + width, [Walc_G123_dict[j][key_i] for j in Walc_list], width=0.1, label=key_i,
                color=color_list[i])
        width += wid

    plt.xlabel('Walc')
    plt.ylabel('avg of G')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.ylim(0, 20)
    plt.legend(fontsize='small')
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_bar_Medu_vs_G(df_maths, df_port, save_path='./plot/bar_Medu_vs_G.png'):
    """
    plot and save figure the bar chart between Medu and G123
    :param df_maths: DataFrame of maths
    :param df_port: DataFrame of portuguese
    :param save_path: save path
    :return:
    """
    Medu_G123_dict = {}
    Medu_list = list(range(5))
    keys_list = ['avgMathsG1','avgMathsG2', 'avgMathsG3', 'avgPortG1', 'avgPortG2', 'avgPortG3']
    for Medu_i in range(5):
      Medu_G123_dict[Medu_i] = {}
      Medu_G123_dict[Medu_i]['avgMathsG1'] = np.round(np.mean(df_maths[df_maths['Medu']==Medu_i]['G1']), 3)
      Medu_G123_dict[Medu_i]['avgMathsG2'] = np.round(np.mean(df_maths[df_maths['Medu']==Medu_i]['G2']), 3)
      Medu_G123_dict[Medu_i]['avgMathsG3'] = np.round(np.mean(df_maths[df_maths['Medu']==Medu_i]['G3']), 3)
      Medu_G123_dict[Medu_i]['avgPortG1'] = np.round(np.mean(df_port[df_port['Medu']==Medu_i]['G1']), 3)
      Medu_G123_dict[Medu_i]['avgPortG2'] = np.round(np.mean(df_port[df_port['Medu']==Medu_i]['G2']), 3)
      Medu_G123_dict[Medu_i]['avgPortG3'] = np.round(np.mean(df_port[df_port['Medu']==Medu_i]['G3']), 3)

    barX = np.arange(5)
    width = 0
    wid = 0.1

    color_list = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for i in range(len(keys_list)):
        key_i = keys_list[i]
        if key_i == 'avgMathsG3':
            lr = LinearRegression()
            lr.fit(np.array(df_maths['Medu']).reshape(-1, 1), np.array(df_maths['G3']))
            plt.plot(barX + width, lr.coef_ * barX + lr.intercept_, color_list[i])
        elif key_i == 'avgPortG3':
            lr = LinearRegression()
            lr.fit(np.array(df_port['Medu']).reshape(-1, 1), np.array(df_port['G3']))
            plt.plot(barX + width, lr.coef_ * barX + lr.intercept_, color_list[i])

        plt.bar(barX + width, [Medu_G123_dict[j][key_i] for j in Medu_list], width=0.1, label=key_i,
                color=color_list[i])
        width += wid

    plt.xlabel('Medu')
    plt.ylabel('avg of G')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.ylim(0, 22)
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_bar_Fedu_vs_G(df_maths, df_port, save_path='./plot/bar_Fedu_vs_G.png'):
    """
    plot and save figure the bar chart between Fedu and G123
    :param df_maths: DataFrame of maths
    :param df_port: DataFrame of portuguese
    :param save_path: save path
    :return:
    """

    Fedu_G123_dict = {}
    Fedu_list = list(range(5))
    keys_list = ['avgMathsG1','avgMathsG2', 'avgMathsG3', 'avgPortG1', 'avgPortG2', 'avgPortG3']
    for Fedu_i in range(5):
      Fedu_G123_dict[Fedu_i] = {}
      Fedu_G123_dict[Fedu_i]['avgMathsG1'] = np.round(np.mean(df_maths[df_maths['Fedu']==Fedu_i]['G1']), 3)
      Fedu_G123_dict[Fedu_i]['avgMathsG2'] = np.round(np.mean(df_maths[df_maths['Fedu']==Fedu_i]['G2']), 3)
      Fedu_G123_dict[Fedu_i]['avgMathsG3'] = np.round(np.mean(df_maths[df_maths['Fedu']==Fedu_i]['G3']), 3)
      Fedu_G123_dict[Fedu_i]['avgPortG1'] = np.round(np.mean(df_port[df_port['Fedu']==Fedu_i]['G1']), 3)
      Fedu_G123_dict[Fedu_i]['avgPortG2'] = np.round(np.mean(df_port[df_port['Fedu']==Fedu_i]['G2']), 3)
      Fedu_G123_dict[Fedu_i]['avgPortG3'] = np.round(np.mean(df_port[df_port['Fedu']==Fedu_i]['G3']), 3)

    barX = np.arange(5)
    width = 0
    wid = 0.1

    color_list = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for i in range(len(keys_list)):
        key_i = keys_list[i]
        if key_i == 'avgMathsG3':
            lr = LinearRegression()
            lr.fit(np.array(df_maths['Fedu']).reshape(-1, 1), np.array(df_maths['G3']))
            plt.plot(barX + width, lr.coef_ * barX + lr.intercept_, color_list[i])
        elif key_i == 'avgPortG3':
            lr = LinearRegression()
            lr.fit(np.array(df_port['Fedu']).reshape(-1, 1), np.array(df_port['G3']))
            plt.plot(barX + width, lr.coef_ * barX + lr.intercept_, color_list[i])
        plt.bar(barX + width, [Fedu_G123_dict[j][key_i] for j in Fedu_list], width=0.1, label=key_i,
                color=color_list[i])
        width += wid

    plt.xlabel('Fedu')
    plt.ylabel('avg of G')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.ylim(0, 22)
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig(save_path)
    plt.show()
    plt.close()




if __name__ == '__main__':
    if not os.path.exists('./plot'):
        os.mkdir('./plot')

    df_maths = pd.read_csv('./data/Maths.csv')
    df_port = pd.read_csv('./data/Portuguese.csv')

    display(df_maths)
    display(df_port)

    X_trainm, X_testm, X_trainp, X_testp = generate_dataset(df_maths, df_port)

    hist_list = ['G3', 'Dalc', 'absences', 'studytime']
    for hist_i in hist_list:
        plot_hist(df_maths, df_port, hist_i, f'./plot/hist_{hist_i}.png')
    plot_heatmap(df_maths, save_path='./plot/maths_heatmap.png')
    plot_heatmap(df_port, save_path='./plot/port_heatmap.png')
    plot_bar_goout_vs_DWalc(df_maths, df_port)
    plot_bar_studytime_vs_G(df_maths, df_port)
    plot_bar_Dalc_vs_G(df_maths, df_port)
    plot_bar_Walc_vs_G(df_maths, df_port)
    plot_bar_Medu_vs_G(df_maths, df_port)
    plot_bar_Fedu_vs_G(df_maths, df_port)


