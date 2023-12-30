import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler


class Clustering:
    def kmeans_clustering(data):
        # Create a copy of the data
        data_copy = data.copy()

        data_number_colum = data_copy.select_dtypes(include=["int", "float"]).columns
        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_number_colum)

        # Select the number of clusters
        n_clusters = st.slider("Chọn số lượng cụm", 1, 10)

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
            fig = px.scatter(X, x=X.iloc[:, 0], y=X.iloc[:, 1], color="cluster")
            st.markdown("Number of Clusters: {}".format(n_clusters))
            st.plotly_chart(fig)

            if len(set(kmeans.labels_)) > 1:  # Only calculate if more than one cluster
                silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
                st.markdown(f"Silhouette Score: {silhouette_avg:.4f}")
            else:
                st.markdown("Silhouette Score not calculated for single cluster or noise.")

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
            fig = px.scatter(X, x=X.iloc[:, 0], y=X.iloc[:, 1], color="cluster")
            st.markdown("Epsilon: {}".format(eps))
            st.markdown("Minimum Samples: {}".format(min_samples))
            st.plotly_chart(fig)

            # Calculate silhouette score (if applicable)
            if len(set(dbscan.labels_)) > 1:  # Only calculate if more than one cluster
                silhouette_avg = silhouette_score(X_scaled, dbscan.labels_)
                st.markdown(f"Silhouette Score: {silhouette_avg:.4f}")
            else:
                st.markdown("Silhouette Score not calculated for single cluster or noise.")

    def mean_shift_clustering(data):

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

            # Estimate bandwidth using a rule of thumb
            bandwidth = estimate_bandwidth(X_scaled)

            # Perform Mean Shift clustering
            ms = MeanShift(bandwidth=bandwidth)
            ms.fit(X_scaled)

            # Add cluster labels to the data
            X["cluster"] = ms.labels_
            X["cluster"] = X["cluster"].astype(str)

            # Visualize the clusters
            fig = px.scatter(X, x=X.iloc[:, 0], y=X.iloc[:, 1], color="cluster")
            st.markdown("Bandwidth: {}".format(bandwidth))
            st.plotly_chart(fig)
