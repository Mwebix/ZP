import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.integrate import simps  # Import the simps function for numerical integration
import io
import base64

# Header with Markdown for styling
st.markdown('<h1 style="color:red;">Webixinol ZA <span style="color:blue;"></span></h1>', unsafe_allow_html=True)

def calculate_z_scores(column_values):
    mean = column_values.mean()
    std_dev = column_values.std()
    z_scores = (column_values - mean) / std_dev
    return z_scores, mean, std_dev

def main():
    st.sidebar.title('Z-Score Analysis')
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx'])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write(df.head())
        
        column_name = st.selectbox("Select column to calculate Z-Scores", df.columns)
        
        if st.button("Calculate Z-Scores"):
            column_values = df[column_name].values
            z_scores, mean, std_dev = calculate_z_scores(column_values)
            df['Z-Scores'] = z_scores
            st.write(df.head())
            
            st.subheader("Histogram of Z-Scores")
            fig, ax = plt.subplots()
            ax.hist(z_scores, bins=20, density=True, alpha=0.6, color='blue', label='Z-Scores')
            
            # Fit a normal distribution curve to the Z-scores
            x = np.linspace(min(z_scores), max(z_scores), 100)
            mu, sigma = norm.fit(z_scores)
            y = norm.pdf(x, mu, sigma)
            ax.plot(x, y, 'r-', label='Normal Distribution')

            # Calculate AUC
            auc = simps(y, x)

            # Display AUC on the plot
            ax.text(0.5, 0.95, f'AUC = {auc:.4f}', horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            
            # Add labels and legend to the plot
            ax.set_title('Normal distribution of Z-scores of {}'.format(column_name))
            ax.set_xlabel('Z-Scores')
            ax.set_ylabel('Density')
            ax.legend()
            
            st.pyplot(fig)
            
            # Download buttons
            st.sidebar.markdown("## Download")
            st.sidebar.markdown(get_table_download_link(df), unsafe_allow_html=True)
            st.sidebar.markdown(get_plot_download_link(fig), unsafe_allow_html=True)

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  
    href = f'<a href="data:file/csv;base64,{b64}" download="z_score_dataframe.csv">Download DataFrame</a>'
    return href

def get_plot_download_link(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    href = f'<a href="data:image/png;base64,{b64}" download="z_score_plot.png">Download Plot</a>'
    return href

if __name__ == "__main__":
    main()
