import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# In this part of visualization, I use "sns.barplot" for categorical variables vs "age". We can see how the rate of change is affected by age.
class fig:
    def __init__(self, main_dataset):
        # Initialize the object with the main dataset
        self.Main_Dataset = main_dataset
        # Filter data for target 0 and target 1
        self.Target_0_data = self.Main_Dataset.loc[self.Main_Dataset["output"] == 0].copy()
        self.Target_0_data.sort_values(by=['age'], inplace=True)
        self.Target_0_data = pd.DataFrame(self.Target_0_data)

        self.Target_1_data = self.Main_Dataset.loc[self.Main_Dataset["output"] == 1].copy()
        self.Target_1_data.sort_values(by=['age'], inplace=True)
        self.Target_1_data = pd.DataFrame(self.Target_1_data)



    def age_cp(self):
        # colors = ["#1593af", "#004280", "#004280", "#004280"]
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=200)
        # sns.barplot(x= self.Target_0_data['age'], y= self.Target_0_data['cp'], errorbar=None,
        #     palette="dark:salmon_r",ax= axes[0]).set(title='Age - CP in Heart Disease = 0')
        # sns.barplot(x= self.Target_1_data['age'], y= self.Target_1_data['cp'], errorbar=None,
        #     palette="dark:salmon_r",ax= axes[1]).set(title='Age - CP in Heart Disease = 1')

        # plt.tight_layout()
        # custom_color = ["#1593af", "#004280", "#004280", "#004280"]

        sns.barplot(x=self.Target_0_data['age'], y=self.Target_0_data['cp'], errorbar=None,
                    palette='coolwarm',ax=axes[0]).set(title='Age - CP in Heart Disease = 0')

        sns.barplot(x=self.Target_1_data['age'], y=self.Target_1_data['cp'], errorbar=None,
                    palette="plasma",ax=axes[1]).set(title='Age - CP in Heart Disease = 1')

        plt.tight_layout()
        st.pyplot(fig)
        # st.warning("Came Here")

    def age_fbs(self):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=200)
        sns.barplot(x= self.Target_0_data['age'], y= self.Target_0_data['fbs'], errorbar=None,
            palette="coolwarm",ax= axes[0]).set(title='Age - fbs in Heart Disease = 0')
        sns.barplot(x= self.Target_1_data['age'], y= self.Target_1_data['fbs'], errorbar=None,
            palette="plasma",ax= axes[1]).set(title='Age - fbs in Heart Disease = 1')
        plt.tight_layout()
        st.pyplot(fig)


    
    def age_rectecg(self):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=200)
        sns.barplot(x= self.Target_0_data['age'], y= self.Target_0_data['restecg'], errorbar=None,
            palette="coolwarm",ax= axes[0]).set(title='Age - restecg in Heart Disease = 0')
        sns.barplot(x= self.Target_1_data['age'], y= self.Target_1_data['restecg'], errorbar=None,
            palette="plasma",ax= axes[1]).set(title='Age - restecg in Heart Disease = 1')
        plt.tight_layout()
        st.pyplot(fig)
    def age_exng(self):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=200)
        sns.barplot(x= self.Target_0_data['age'], y= self.Target_0_data['exng'], errorbar=None,
            palette="coolwarm",ax= axes[0]).set(title='Age - exng in Heart Disease = 0')
        sns.barplot(x= self.Target_1_data['age'], y= self.Target_1_data['exng'], errorbar=None,
            palette="plasma",ax= axes[1]).set(title='Age - exng in Heart Disease = 1')
        plt.tight_layout()
        st.pyplot(fig)

    def age_slp(self):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=200)
        sns.barplot(x= self.Target_0_data['age'], y= self.Target_0_data['slp'], errorbar=None,
            palette="coolwarm",ax= axes[0]).set(title='Age - slp in Heart Disease = 0')
        sns.barplot(x= self.Target_1_data['age'], y= self.Target_1_data['slp'], errorbar=None,
            palette="plasma",ax= axes[1]).set(title='Age - slp in Heart Disease = 1')
        plt.tight_layout()
        st.pyplot(fig)
    def age_caa(self):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=200)
        sns.barplot(x= self.Target_0_data['age'], y= self.Target_0_data['caa'], errorbar=None,
            palette="coolwarm",ax= axes[0]).set(title='Age - caa in Heart Disease = 0')
        sns.barplot(x= self.Target_1_data['age'], y= self.Target_1_data['caa'], errorbar=None,
            palette="plasma",ax= axes[1]).set(title='Age - caa in Heart Disease = 1')
        plt.tight_layout()
        st.pyplot(fig)
    def age_thall(self):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=200)
        sns.barplot(x= self.Target_0_data['age'], y= self.Target_0_data['thall'], errorbar=None,
            palette="coolwarm",ax= axes[0]).set(title='Age - thall in Heart Disease = 0')
        sns.barplot(x= self.Target_1_data['age'], y= self.Target_1_data['thall'], errorbar=None,
            palette="plasma",ax= axes[1]).set(title='Age - thall in Heart Disease = 1')
        plt.tight_layout()
        st.pyplot(fig)

    def KnnPlot(self,range_k,training_acc_1,test_acc_1):
        plt.figure(figsize=(15,5), dpi=200)    
        plt.plot(range_k, training_acc_1, label='Acc of training', color= 'black')
        plt.plot(range_k, test_acc_1, label='Acc of test set', color= '#E72B3B')
        plt.ylabel('Acc')
        plt.xlabel('Number of Neighbors')
        plt.title('Acc - Number of K')
        plt.legend()
        plt.xticks(range(1,20))
        plt.annotate('Best K_neighbor', xy=(3,0.89),xytext=(7.2,0.86), arrowprops=dict(facecolor='#E72B3B', shrink=0.05),fontsize=20)
        plt.axvline(x = 3, linestyle= 'dotted', c= 'black')
        # plt.show()
        st.pyplot(plt)
    
    def KnnConfusionMatrix(self,conf_matrix_1,p):
        # colors = ["black", "#E72B3B", "#E72B3B", "#E72B3B"]
        # colors = ["#87CEEB", "#9370DB", "#6A5ACD", "#483D8B"]
        # colors = ["#1593af", "#004280", "#1593af", "#004280"]
        colors = ["#1593af", "#004280", "#004280", "#004280"]


        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        fig = plt.figure(figsize=(15, 3), dpi=200)
        ax = plt.subplot()
        plt.title(f"Confusion_matrix , KNN , K = 3 , P = {p}")
        annot = np.array([[f"{conf_matrix_1[0, 0]}", f"{conf_matrix_1[0, 1]}"],
                        [f"{conf_matrix_1[1, 0]}", f"{conf_matrix_1[1, 1]}"]], dtype=object)


        sns.heatmap(conf_matrix_1,
                    annot=annot,
                    annot_kws={"size": 11},
                    ax=ax,
                    fmt='',
                    cmap=cmap,
                    cbar=True,
                    )
        plt.xlabel("Pred")
        plt.ylabel("Real")
        # plt.show()
        st.pyplot(fig)

    def SvmPlot(self,training_acc,test_acc,type,C,xy,xytext,x):
        plt.figure(figsize=(15,5), dpi=200)    
        plt.plot(C, training_acc, label='Acc of training', color= 'black')
        plt.plot(C, test_acc, label='Acc of test set', color= '#E72B3B')
        plt.ylabel('Acc')
        plt.xlabel('Number of C')
        plt.title(f'Acc - Number of C, {type}')
        plt.legend()
        plt.xticks(range(0,50))
        plt.annotate('Best C', xy=xy,xytext=xytext, arrowprops=dict(facecolor='#E72B3B', shrink=0.05),fontsize=20)
        plt.axvline(x = x, linestyle= 'dotted', c= 'black')
        # plt.show()
        st.pyplot(plt)
    def SvmConfusionMatrix(self,conf_matrix_3,type,c):
        # colors = ["black", "#E72B3B", "#E72B3B", "#E72B3B"]
        colors = ["#1593af", "#004280", "#004280", "#004280"]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        fig = plt.figure(figsize=(15, 3), dpi=200)
        ax = plt.subplot()
        plt.title(f"Confusion_matrix (SVM , {type} , C = {c})")
        annot = np.array([[f"{conf_matrix_3[0, 0]}", f"{conf_matrix_3[0, 1]}"],
                        [f"{conf_matrix_3[1, 0]}", f"{conf_matrix_3[1, 1]}"]], dtype=object)

        sns.heatmap(conf_matrix_3,
                    annot=annot,
                    annot_kws={"size": 11},
                    ax=ax,
                    fmt='',
                    cmap=cmap,
                    cbar=True,
                    )
        plt.xlabel("Pred")
        plt.ylabel("Real")
        st.pyplot(fig)