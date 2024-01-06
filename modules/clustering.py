import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


class Clustering:
    def kmeans_clustering(data):
        # Create a copy of the data
        data_copy = data.copy()

        data_number_colum = data_copy.select_dtypes(include=["int", "float"]).columns
        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_number_colum)

        # Select the number of clusters
        n_clusters = st.slider("Chọn số lượng cụm", 2, 10)

        # Get the data for clustering
        X = data_copy[feature_columns]

        # Standardize the data before clustering

        # Perform K-Means clustering
        if not feature_columns:
            st.warning("Chon cot tinh nang")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_scaled)

            # Add cluster labels to the data
            X["cluster"] = kmeans.labels_
            X["cluster"] = X["cluster"].astype(str)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Clustering Visualize #####")
                if len(feature_columns) <= 2:
                    fig = px.scatter(X, x=X.iloc[:, 0], y=X.iloc[:, 1], color="cluster")
                    st.markdown("Number of Clusters: {}".format(n_clusters))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    pass

            with col2:

                st.markdown("##### Clustering Result Visualize #####")
                silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
                st.markdown(f"Silhouette Score: {silhouette_avg:.4f}")
                st.dataframe(X, use_container_width=True)

    def dbscan_clustering(data):

        # Create a copy of the data
        data_copy = data.copy()

        # Select numerical feature columns
        data_number_colum = data_copy.select_dtypes(include=["int", "float"]).columns
        feature_columns = st.multiselect("Chọn biến tính năng", data_number_colum)

        # Get the data for clustering
        X = data_copy[feature_columns]

        # Standardize the data before clustering
        if not feature_columns:
            st.warning("Chon cot tinh nang")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Set DBSCAN hyperparameters
            eps = st.slider("Chọn epsilon (khoảng cách)", 0.1, 1.0, value=0.5)  # Adjust default value as needed
            min_samples = st.slider("Chọn số lượng điểm tối thiểu", 5, 20,
                                    value=10)  # Adjust default value as needed

            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(X_scaled)

            # Add cluster labels to the data
            X["cluster"] = dbscan.labels_
            X["cluster"] = X["cluster"].astype(str)

            # Visualize the clusters
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Clustering Visualize #####")
                if len(feature_columns) <= 2:
                    fig = px.scatter(X, x=X.iloc[:, 0], y=X.iloc[:, 1], color="cluster")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    pass

            with col2:
                st.markdown("##### Clustering Result Visualize #####")
                silhouette_avg = silhouette_score(X_scaled, dbscan.labels_)
                st.markdown(f"Silhouette Score: {silhouette_avg:.4f}")
                st.dataframe(X, use_container_width=True)

    def optics_clustering(data):
        # Create a copy of the data
        data_copy = data.copy()

        # Select numerical feature columns
        data_number_colum = data_copy.select_dtypes(include=["int", "float"]).columns
        feature_columns = st.multiselect("Chọn biến tính năng", data_number_colum)

        # Get the data for clustering
        X = data_copy[feature_columns]
        if not feature_columns:
            st.warning("Chon cot tinh nang")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Set OPTICS hyperparameters
            # Adjust default value as needed
            min_samples = st.slider("Chọn số lượng điểm tối thiểu", 5, 20, value=10)  # Adjust default value as needed

            # Perform OPTICS clustering
            optics = OPTICS(min_samples=min_samples, metric='euclidean')
            optics.fit(X_scaled)

            # Extract DBSCAN-like clusters from the OPTICS output

            # Add cluster labels to the data
            X["cluster"] = optics.labels_
            X["cluster"] = X["cluster"].astype(str)

            # Visualize the clusters
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Clustering Visualize #####")
                if len(feature_columns) <= 2:
                    fig = px.scatter(X, x=X.iloc[:, 0], y=X.iloc[:, 1], color="cluster")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    pass

            with col2:
                st.markdown("##### Clustering Result Visualize #####")
                silhouette_avg = silhouette_score(X_scaled, optics.labels_)
                st.markdown(f"Silhouette Score: {silhouette_avg:.4f}")
                st.dataframe(X, use_container_width=True)


