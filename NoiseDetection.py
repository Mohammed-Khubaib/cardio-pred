# We can use boxplots in continuous variables for detect any noises.
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
def boxplots(Main_Dataset):
    Numerical = ['age', 'trtbps','chol','thalachh','oldpeak']
    i = 0
    while i < 5:
        fig = plt.figure(figsize = [30,3], dpi=200)
        plt.subplot(2,2,1)
        sns.boxplot(x = Numerical[i], data = Main_Dataset,
            boxprops = dict(facecolor = "#E72B3B"))
        st.pyplot(fig)
        i += 1

def NoiseRemoved(Main_Dataset):
    fig = plt.figure(figsize = [15,3], dpi=200)
    sns.boxplot(x = 'chol', data = Main_Dataset,
        boxprops = dict(facecolor = "#E72B3B"))
    st.pyplot(fig) 