from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import os
import pandas as pd 
from math import ceil

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/form')
def form():
    return render_template('testform.html')

from flask import request  # Import request from Flask

from math import ceil

@app.route('/data')
def data():
    # Menentukan path lengkap file stroke.csv dan transformasi.csv
    csv_stroke_path = os.path.join('stroke-dataset.csv')
    csv_transformasi_path = os.path.join('transformasi.csv')
    
    # Membaca dataset stroke.csv dan transformasi.csv
    df_stroke = pd.read_csv(csv_stroke_path)
    df_transformasi = pd.read_csv(csv_transformasi_path)

    # Mendapatkan nomor halaman dari parameter query jika ada, jika tidak, gunakan 1 sebagai halaman default
    page_num = int(request.args.get('page', 1))

    # Menentukan jumlah data per halaman
    data_per_page = 500

    # Menghitung indeks awal dan akhir untuk data pada halaman yang diminta
    start_idx = (page_num - 1) * data_per_page
    end_idx = start_idx + data_per_page

    # Memotong DataFrame sesuai dengan halaman yang diminta
    df_stroke_page = df_stroke.iloc[start_idx:end_idx]
    df_transformasi_page = df_transformasi.iloc[start_idx:end_idx]

    # Menghitung jumlah total halaman untuk pagination data transformasi
    total_rows_transformasi = len(df_transformasi)
    total_pages_transformasi = ceil(total_rows_transformasi / data_per_page)

    return render_template('data.html', df_stroke=df_stroke_page, df_transformasi=df_transformasi_page, 
                           css_file='css/sb-admin-2', total_pages_transformasi=total_pages_transformasi, 
                           current_page=page_num)


@app.route('/smote')
def smote():
    return render_template('smote.html')

@app.route('/report')
def report():
    return render_template('report.html')

app.run(debug=True)