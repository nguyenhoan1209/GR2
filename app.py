import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from streamlit_option_menu import option_menu
from streamlit_timeline import timeline
import numpy as np
import math
import os
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.stats import t
from modules import Chart, Info, Regression, Classification, Clustering

st.set_page_config("DataApp", layout="wide", initial_sidebar_state="expanded", )


def footer():
    st.markdown(
        """
        <head>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        </head>
        <style>
            footer
            {
            visibility:hidden;
            }
            .a {
                
                background-color: #f0f2f6;
                padding: 20px;
                text-align: center;
            }
            
            .icon-list {
                display: flex;
                justify-content: center;
                align-items: center;
            }

            .icon-list-item {
                margin: 10px;
                text-align: center;
                cursor: pointer;
            }

            .icon-list-item i {
                display: block;
                font-size: 20px;
                color: black;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="a">
            <h6>Liên hệ với tôi</h6>
            <div class="icon-list">
                <div class="icon-list-item">
                    <a href="https://github.com" target="_blank">
                        <i class="fab fa-github"></i>
                    </a>
                </div>
                <div class="icon-list-item">
                    <a href="https://twitter.com" target="_blank">
                        <i class="fab fa-twitter"></i>
                    </a>
                </div>
                <div class="icon-list-item">
                    <a href="https://youtube.com" target="_blank">
                        <i class="fab fa-youtube"></i>
                    </a>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def search():
    st.markdown("""
            <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
            <style>
                body {
                font-family: Sans serif;
                }

                * {
                box-sizing: border-box;
                }

                form.example input[type=text] {
                padding: 10px;
                font-size: 17px;
                border: 2px solid white;
                float: left;
                width: 700px;  /* Set width to 100px */
                background: #f1f1f1;
                border-radius: 15px;
                }

                form.example button {
                float: left;
                width: 100px;  /* Set width to auto to adjust based on content */
                padding: 10px;
                background: #FF4B4B;
                color: white;
                font-size: 17px;
                border: 2px solid white;
                border-left: none;
                cursor: pointer;
                border-radius: 15px;
                }

                form.example button:hover {
                background: #FF4B4B;
                }

                form.example::after {
                content: "";
                clear: both;
                display: table;
                }
            </style>
            </head>
            <body>
            """,
                unsafe_allow_html=True)

    st.markdown("""
            <form class="example" action="" style="margin:auto;max-width:800px">
            <input type="text" placeholder="Search.." name="search2">
            <button type="submit"><i class="fa fa-search"></i></button>
            </form>""",
                unsafe_allow_html=True)
    st.markdown(
        """
                <head>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
                </head>
                """,
        unsafe_allow_html=True
    )


@st.cache_data
def load_data(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    if file_extension == '.csv':
        return pd.read_csv(file)
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(file)


def summary(df):
    summary = df.describe()
    return summary


def summary_p(df):
    summary = df.describe()
    return summary


### analyze_data      
def analyze_data(data):
    # Perform basic data analysis
    st.write(" # Data Analysis # ")
    st.write("#### Dữ liệu ####")
    st.write("Data")
    with st.expander("See data", expanded=True):
        edited_df = st.data_editor(data,use_container_width=True,num_rows="dynamic")

    st.markdown("---")
    ######
    st.write("#### Thống kê mô tả một chiều ####")

    st.markdown("###### Bảng giá trị thống kê mô tả ######")
    use_sample_stats = st.checkbox('Hiệu chỉnh mẫu thống kê', value=True)
    if use_sample_stats:
        # compute and show the sample statistics
        st.dataframe(summary(edited_df), use_container_width=True)
        st.download_button(
            label="Download data as CSV",
            data=summary(data).to_csv(index=False),
            file_name='data_analyze.csv',
            mime='text/csv')

    else:
        # compute and show the population statistics
        st.dataframe(summary_p(edited_df), use_container_width=True)
        st.download_button(
            label="Download data as CSV",
            data=summary_p(data).to_csv(index=False),
            file_name='data_analyze.csv',
            mime='text/csv')
    footer()


#### Data visualization

#### hypothesis test

# main function
def main():
    with st.sidebar:
        st.sidebar.markdown("---")
        st.markdown("#### Chọn chức năng ####")
        selected = option_menu(None, ["Dữ liệu", "Thống kê", "Trực quan hóa", "Hồi quy", "Phân lớp", "Phân cụm"],
                               icons=['clipboard-data', 'table', "bar-chart-fill", 'rulers', 'diamond-half'],
                               menu_icon="cast", default_index=0, styles={
                "st": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "black", "font-size": "15px"},
                "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px",
                             "--hover-color": "#eee"},
            })

    with st.container():
        with st.sidebar:
            st.sidebar.markdown("---")
            st.markdown("#### Tải lên dữ liệu ####")
            file = st.file_uploader("", type=["csv", "xlsx", "xls"])

        if file is not None:

            data = load_data(file)

            if selected == 'Dữ liệu':
                search()
                Info.info(data)

            if selected == 'Thống kê':
                search()
                analyze_data(data)

            if selected == 'Trực quan hóa':
                search()
                st.write(" # Trực quan hóa dữ liệu # ")
                st.write("#### Dữ liệu ####")
                st.write("Data")
                edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
                st.markdown("---")
                st.write("#### Chọn loại biểu đồ ####")
                chart_type = st.selectbox("", ["Bar", "Line", "Scatter", "Pie", "Boxplot", "Heatmap"])

                Chart.create_chart(chart_type, edit_data)

            if selected == 'Hồi quy':
                search()

                st.write(" # Hồi quy tuyến tính # ")
                st.write("#### Dữ liệu ####")
                st.write("Data")
                edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
                st.markdown("---")
                regression_type = st.selectbox("", ["OLS Linear Regression", 'Ridge', 'Lasso'])
                if regression_type == "OLS Linear Regression":
                    Regression.simple_linear_regresstion(data)
                if regression_type == "Ridge":
                    Regression.ridge_regression(data)
                if regression_type == "Lasso":
                    Regression.lasso_regression(data)

            if selected == 'Phân lớp':
                search()
                st.write(" # Phân lớp # ")
                st.write("#### Dữ liệu ####")
                st.write("Data")
                edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
                st.markdown("---")
                class_type = st.selectbox("", ["KNN", 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM'])
                if class_type == 'KNN':
                    Classification.knn_classification(data)
                if class_type == 'Logistic Regression':
                    Classification.lgreg_classification(data)
                if class_type == 'Random Forest':
                    Classification.randomfor_classification(data)
                if class_type == 'Naive Bayes':
                    Classification.naivebayes_classification(data)
                if class_type == 'SVM':
                    Classification.svm_classification(data)

            if selected == 'Phân cụm':
                search()
                st.write(" # Phân cụm # ")
                st.write("#### Dữ liệu ####")
                st.write("Data")
                edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
                st.markdown("---")
                class_type = st.selectbox("", ["K Means", 'DBSCAN', 'Mean Shift'])
                if class_type == 'K Means':
                    Clustering.kmeans_clustering(data)
                if class_type == 'DBSCAN':
                    Clustering.dbscan_clustering(data)
                if class_type == 'Mean Shift':
                    Clustering.mean_shift_clustering(data)

            # if selected =='Kiểm định':
            #     search()
            #     st.write(" # Kiểm định giả thuyết thống kê # ")
            #     st.write("#### Dữ liệu ####")
            #     st.write("Data")
            #     edit_data= st.data_editor(data,use_st_width=True,num_rows="dynamic")
            #     st.markdown("---")
            #     st.write("#### Chọn phương thức muốn kiểm định ####")
            #     test_type = st.selectbox("", [None,"Kiểm định một mẫu", "Kiểm định nhiều mẫu", "Kiểm định phi tham số"])
            #     hypothesis_test(test_type, edit_data)
        else:
            st.balloons()
            st.markdown(
                """
        <h1>Trang chủ</h1>
        """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
