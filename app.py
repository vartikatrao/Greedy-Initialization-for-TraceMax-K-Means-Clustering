import pickle
from flask import Flask, render_template

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend (for PNGs) instead of the default interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import random as rand
app = Flask(__name__,template_folder='Templates')
from flask import request
import time 
def init_centroids (all_vals, K):
  centroids=[]
  centroids_idx = rand.sample(range(0, len(all_vals)), K)
  for i in centroids_idx:
    centroids.append(np.array(all_vals[i]))
  return centroids
def cluster_data (all_vals, centroids):
  assignments = []
  for data_point in all_vals:
    dist_point_clusts= []
    for centroid in centroids:
      dist_btw_point_cluster= np.linalg.norm(np.array(data_point)- np.array(centroid))
      dist_point_clusts.append(dist_btw_point_cluster)
    assignment = np.argmin(dist_point_clusts)
    assignments.append(assignment)
  return assignments
def new_centroids (all_vals, centroids, assignments, K):
  new_centroids =[]
  for i in range (K):
    pt_cluster =[]
    for x in range (len(all_vals)):
      if (assignments[x]==i):
        pt_cluster.append(all_vals[x])
    mean_c= np.mean (pt_cluster, axis=0)
    new_centroids.append(mean_c)
  return new_centroids
def kmeans (all_vals, K, max_iteration=100):
  it=0
  assignments = []
  centroids = init_centroids(all_vals, K)
  while (it<=max_iteration):
    it+=1
    assignments = cluster_data(all_vals, centroids)
    new_c= new_centroids(all_vals, centroids, assignments, K)
    if (all(np.array_equal(x, y) for x, y in zip(new_c, centroids ))):
      break
    centroids= new_c

  return (assignments, centroids, it)

def kmeans_with_new_formulation(X, k):
    n = X.shape[1]
    # Initialization of F
    label = np.ceil(k * np.random.rand(n)).astype(int)  # random initialization
    F = csr_matrix((np.ones(n), (np.arange(n), label - 1)), shape=(n, k)).toarray()
    
    start_time = time.time()
    A = X.T @ X
    s = np.ones(k)
    M = np.ones((n, k))
    
    for iter in range(1000):
        label_last = label.copy()
        #updating vector s based on current values of F
        for i in range(k):
            f = F[:, i]
            temp1 = f.T @ A @ f
            temp2 = f.T @ f
            s_i = np.sqrt(temp1) / (temp2)
            s[i] = s_i
        #updating M based on current values of F
        for iter_re in range(100):
            for j in range(k):
                f = F[:, j]
                temp4 = A @ f
                temp3 = np.sqrt(f.T @ temp4)
                m_j = (1 / temp3) * temp4
                M[:, j] = m_j
            #updates F based on calculated values of M and s
            temp_s = s
            S = np.tile(temp_s, (n, 1))
            temp_M = 2 * S * M
            temp_S = S ** 2
            temp5 = temp_S - temp_M
            label_new = np.argmin(temp5, axis=1) + 1
            F = csr_matrix((np.ones(n), (np.arange(n), label_new - 1)), shape=(n, k)).toarray()
            #convergence check
            if np.array_equal(label, label_new):
                break
            label = label_new
        if np.array_equal(label, label_last):
            iter_num = iter
            break
    time_elapsed = time.time() - start_time
    center = X @ (F @ np.diag(1.0 / F.sum(axis=0)))  # = XF((F.T@F)-1)
    temp6 = X - center @ F.T
    obj = np.linalg.norm(temp6, 'fro') ** 2
    
    return label, iter_num, center, obj

def max_dist(data1, X):
    n = X.shape[1]
    max_dist = 0
    data2 = np.zeros(X.shape[0])
    for i in range(n):
        dist = np.linalg.norm(X[:, i] - data1)
        if dist > max_dist:
            max_dist = dist
            data2 = X[:, i]
    return data2

def initialization(X, k, centers):
    mn = np.mean(X, axis=1)
    data1 = mn
    prev = mn
    data2 = max_dist(data1, X)
    while not np.array_equal(data1, prev):
        prev = data1
        data1 = data2
        data2 = max_dist(data1, X)
    centers.append(data1)
    centers.append(data2)
    for i in range(2, k):
        mn_centers= np.mean(centers, axis=0)
        data = max_dist(mn_centers, X)
        centers.append(data)
    return centers

def indicator_matrix(X, centers):
    n = X.shape[1]
    k = len(centers)
    label = np.zeros(n)
    F = np.zeros((n, k))
    for i in range(n):
        dist = np.linalg.norm(X[:, i] - centers[0])
        label[i] = 0
        for j in range(1, k):
            if np.linalg.norm(X[:, i] - centers[j]) < dist:
                dist = np.linalg.norm(X[:, i] - centers[j])
                label[i] = j
    for i in range(n):
        F[i, int(label[i])] = 1
    return F
def modified_kmeans_with_new_formulation(X, k):
    n = X.shape[1]
    initial_centers = initialization(X, k, [])
    F = indicator_matrix(X, initial_centers)
    label = np.argmax(F, axis=1)
    start_time = time.time()
    A = X.T @ X
    s = np.ones(k)
    M = np.ones((n, k))
    for iter in range(1000):
        label_last = label.copy()
        for i in range(k):
            f = F[:, i]
            temp1 = f.T @ A @ f
            temp2 = f.T @ f
            s_i = np.sqrt(temp1) / temp2
            s[i] = s_i
        for iter_re in range(100):
            for j in range(k):
                f = F[:, j]
                temp4 = A @ f
                temp3 = np.sqrt(f.T @ temp4)
                m_j = np.true_divide(temp4, temp3 + np.finfo(float).eps)
                M[:, j] = m_j
            temp_s = s
            S = np.tile(temp_s, (n, 1))
            temp_M = 2 * S * M
            temp_S = S ** 2
            temp5 = temp_S - temp_M
            label_new = np.argmin(temp5, axis=1) + 1
            F = csr_matrix((np.ones(n), (np.arange(n), label_new - 1)), shape=(n, k)).toarray()
            if np.array_equal(label, label_new):
                break
            label = label_new
        if np.array_equal(label, label_last):
            iter_num = iter
            break
    else:
        iter_num = 1000
    time_elapsed = time.time() - start_time
    center = X @ (F @ np.diag(1.0 / F.sum(axis=0)))
    temp6 = X - center @ F.T
    obj = np.linalg.norm(temp6, 'fro') ** 2
    return label, iter_num, center, obj

#after k means make a function to calculate the sum of squared distances from each point to the center of the assigned cluster 
def sum_squared_distances_kmeans_fn(X, centers, label, n):
    k = len(centers)
    sum = 0
    for i in range(n):
        sum += np.linalg.norm(X[i] - centers[int(label[i])]) ** 2
    return sum

def sum_squared_distances(X, centers, label):
    print("X", X)
    print("centers", centers)
    print("label", label)
    n= len(X)
    k= len(centers)
    sum=0
    print(max(label))
    for i in range(n):
        sum+= np.linalg.norm(X[i]- centers[int(label[i])-1])**2
    return sum

def calculate_kmeans(X, num_clusters):
        label_kmeans_with_new_formulation, iter_num_with_new_formulation, center_with_new_formulation, obj_with_new_formulation = kmeans_with_new_formulation(X.T, num_clusters)
        label_modified_kmeans_with_new_formulation, iter_num_modified_with_new_formulation, center_modified_with_new_formulation, obj_modified_with_new_formulation = modified_kmeans_with_new_formulation(X.T, num_clusters)
        label_kmeans, centers_kmeans, it_kmeans = kmeans(X, num_clusters)
        sum_squared_distances_kmeans = sum_squared_distances_kmeans_fn(X, centers_kmeans, label_kmeans, len(X))
        print(X)
        print(centers_kmeans)
        print(sum_squared_distances_kmeans)
        sum_squared_distances_kmeans_with_new_formulation = sum_squared_distances(X, center_with_new_formulation.T, label_kmeans_with_new_formulation)
        sum_squared_distances_modified_kmeans_with_new_formulation = sum_squared_distances(X, center_modified_with_new_formulation.T, label_modified_kmeans_with_new_formulation)
        fig_kmeans, ax = plt.subplots(figsize=(10, 8))
        print("center_kmeans", centers_kmeans)
        print("2", center_with_new_formulation)
        print("3", center_modified_with_new_formulation)
        fig_kmeans, ax_kmeans = plt.subplots(figsize=(10, 8))
        ax_kmeans.scatter(X.T[0, :], X.T[1, :], c= label_kmeans, cmap='viridis', marker= 'o', edgecolors='w', s=50, label= 'Data Points')
        ax_kmeans.scatter(np.array(centers_kmeans)[:, 0], np.array(centers_kmeans)[:, 1], c= 'red', marker= 'x', s=200, label= 'Cluster Centers')
        ax_kmeans.set_title("Lloyd's Algorithm", fontsize= 28, color= "white")
        ax_kmeans.tick_params(axis='x', colors='white', labelsize=16)
        ax_kmeans.tick_params(axis='y', colors='white', labelsize=16)
        ax_kmeans.grid(False)
        ax_kmeans.legend()
        fig_kmeans.patch.set_alpha(0)
        buf_kmeans = BytesIO()
        plt.savefig(buf_kmeans, format='png')
        plt.close()
        data_kmeans = base64.b64encode(buf_kmeans.getbuffer()).decode('ascii')

        fig_kmeans_with_new_formulation, ax_kmeans_with_new_formulation = plt.subplots(figsize=(10, 8))
        ax_kmeans_with_new_formulation.scatter(X.T[0, :], X.T[1, :], c=label_kmeans_with_new_formulation, cmap='viridis', marker='o', edgecolors='w', s=50, label='Data Points')
        ax_kmeans_with_new_formulation.scatter(center_with_new_formulation[0][:], center_with_new_formulation[1][:], c='red', marker='x', s=200, label='Cluster Centers')
        ax_kmeans_with_new_formulation.set_title("K-means with New Formulation", fontsize=28, color="white")
        ax_kmeans_with_new_formulation.tick_params(axis='x', colors='white', labelsize=16)
        ax_kmeans_with_new_formulation.tick_params(axis='y', colors='white', labelsize=16)
        ax_kmeans_with_new_formulation.grid(False)
        ax_kmeans_with_new_formulation.legend()
        fig_kmeans_with_new_formulation.patch.set_alpha(0)
        buf_kmeans_with_new_formulation = BytesIO()
        plt.savefig(buf_kmeans_with_new_formulation, format='png')
        plt.close()
        data_kmeans_with_new_formulation = base64.b64encode(buf_kmeans_with_new_formulation.getbuffer()).decode('ascii')

        fig_modified_kmeans_with_new_formulation, ax_modified_kmeans_with_new_formulation = plt.subplots(figsize=(10, 8))
        ax_modified_kmeans_with_new_formulation.scatter(X.T[0, :], X.T[1, :], c=label_modified_kmeans_with_new_formulation, cmap='viridis', marker='o', edgecolors='w', s=50, label='Data Points')
        ax_modified_kmeans_with_new_formulation.scatter(center_modified_with_new_formulation[0][:], center_modified_with_new_formulation[1][:], c='red', marker='x', s=200, label='Cluster Centers')
        ax_modified_kmeans_with_new_formulation.set_title("Modified K-means with New Formulation", fontsize=28, color="white")
        ax_modified_kmeans_with_new_formulation.tick_params(axis='x', colors='white', labelsize=16)
        ax_modified_kmeans_with_new_formulation.tick_params(axis='y', colors='white', labelsize=16)
        ax_modified_kmeans_with_new_formulation.grid(False)
        ax_modified_kmeans_with_new_formulation.legend()
        fig_modified_kmeans_with_new_formulation.patch.set_alpha(0)
        buf_modified_kmeans_with_new_formulation = BytesIO()
        plt.savefig(buf_modified_kmeans_with_new_formulation, format='png')
        plt.close()
        data_modified_kmeans_with_new_formulation = base64.b64encode(buf_modified_kmeans_with_new_formulation.getbuffer()).decode('ascii')
        return render_template('result.html', data1= data_kmeans, data2= data_kmeans_with_new_formulation, data3= data_modified_kmeans_with_new_formulation, sum_squared_distances_kmeans= sum_squared_distances_kmeans, sum_squared_distances_kmeans_with_new_formulation= sum_squared_distances_kmeans_with_new_formulation, sum_squared_distances_modified_kmeans_with_new_formulation= sum_squared_distances_modified_kmeans_with_new_formulation, n_iterations_kmeans= it_kmeans, n_iterations_kmeans_with_new_formulation= iter_num_with_new_formulation, n_iterations_modified_kmeans_with_new_formulation= iter_num_modified_with_new_formulation)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/generate_random')
def generate_random():
    return render_template('generate_random.html')

@app.route('/iris_dataset')
def iris_dataset():
    data= pd.read_csv('Datasets//IRIS.csv')
    X=data.iloc[:,[0,1,2,3]].values
    Y=data.iloc[:,4].values
    label_encoder = LabelEncoder()
    data['numeric_labels'] = label_encoder.fit_transform(Y)
    return calculate_kmeans(X,3)

@app.route('/balance_dataset')
def balance_dataset():
    data= pd.read_csv('Datasets//balance-scale.csv')
    X = data.drop(['Class'], axis = 1).values
    return calculate_kmeans(X,3)
    
@app.route('/dermatology_dataset')
def dermatology_dataset():
    data= pd.read_csv('Datasets//dermatology_database_1.csv')
    X= data.drop(['class'], axis = 1)
    X= X.drop(['age'], axis = 1)
    X= X.values
    return calculate_kmeans(X,6)

@app.route('/ecoli_dataset')
def ecoli_dataset():
    data= pd.read_csv('Datasets//ecoli.csv')
    X= data.drop(['SITE'], axis = 1)
    X= X.drop(['SEQUENCE_NAME'], axis = 1)
    X= X.values
    return calculate_kmeans(X,3)

@app.route('/predict_random', methods=['GET', 'POST'])
def predict_random():
    X=np.array([])
    k=0
    if request.method == 'POST':
        n = int(request.form['number1'])
        dim= int(request.form['number2'])
        k = int(request.form['number3'])
        X = np.random.rand(n, dim)
        return calculate_kmeans(X, k)
    else: 
        render_template ('result.html')
    
    

if __name__ == '__main__':
    app.run(debug=True)