from flask import Flask,render_template,flash, redirect,url_for,session,logging,request,jsonify
import sqlite3
from flask_cors import CORS
from flask import send_file
from io import BytesIO
from flask_wtf.file import FileField
from wtforms import SubmitField
from flask_wtf import FlaskForm

from nonSpark import linreg1_nonspark, linreg2_nonspark, savepkl1, savepkl2
from useSpark import linreg1_spark, linreg2_spark


app = Flask(__name__, static_folder='static')
# run_with_ngrok(app)  # Start ngrok when app is run
# CORS(app)
CORS(app, resources=r'/api/*')

# app.debug = False
app.secret_key = 'key_big_data_app'

@app.route("/")
def index():
    return render_template("index.html")
    # return "Hello Brother"

@app.route("/login",methods=["GET", "POST"])
def login():
  conn = connect_db()
  db = conn.cursor()
  #conn = sqlite3.connect('fga_big_data_rev2.db')
  #db = conn.cursor()
  msg = ""
  if request.method == "POST":
      mail = request.form["mail"]
      passw = request.form["passw"]

      rs = db.execute("SELECT * FROM user WHERE Mail=\'"+ mail +"\'"+" AND Password=\'"+ passw+"\'" + " LIMIT 1")

      conn.commit()

      hasil = []
      for v_login in rs:
         hasil.append(v_login)

      if hasil:
        session['name'] = v_login[3]
        session['mail'] = v_login[1]
        return redirect(url_for("bigdataApps"))
      else:
        msg = "Masukkan Username (Email) dan Password dgn Benar!"

  return render_template("login.html", msg = msg)

@app.route("/register", methods=["GET", "POST"])
def register():
  conn = connect_db()
  db = conn.cursor()
  if request.method == "POST":
      mail = request.form['mail']
      uname = request.form['uname']
      passw = request.form['passw']

      cmd = "insert into user(Mail, Password,Name,Level) values('{}','{}','{}','{}')".format(mail,passw,uname,'1')
      conn.execute(cmd)
      conn.commit()

      # conn = db

      return redirect(url_for("login"))
  return render_template("register.html")


@app.route("/linreg1_nonspark", methods=["GET", "POST"])
def lr1_nonspark():
  mydict = dict(linreg1_nonspark())
  title = 'Linear regression with simulation data using SKLearn'
  return render_template('MybigdataApps.html', pname = title, y_aktual = mydict["y_aktual"], y_prediksi = mydict["y_prediksi"], mape = mydict["mape"])

@app.route("/linreg2_nonspark", methods=["GET", "POST"])
def lr2_nonspark():
  mydict = dict(linreg2_nonspark())
  title = 'Linear regression with addjusted simulation data using SKLearn'
  return render_template('MybigdataApps.html', pname = title, y_aktual = mydict["y_aktual"], y_prediksi = mydict["y_prediksi"], mape = mydict["mape"])

@app.route("/fp_3_1_nonspark", methods=["GET", "POST"])
def fp_3_1_nonspark():
  mydict = dict(savepkl1())
  title = 'Linear regression with addjusted simulation data using SKLearn'
  return render_template('MybigdataApps.html', pname = title, y_aktual = mydict["y_aktual"], y_prediksi = mydict["y_prediksi"], mape = mydict["mape"])

@app.route("/fp_3_2_nonspark", methods=["GET", "POST"])
def fp_3_2_nonspark():
  mydict = dict(savepkl2())
  title = 'Linear regression with addjusted simulation data using SKLearn'
  return render_template('MybigdataApps.html', pname = title, y_aktual = mydict["y_aktual"], y_prediksi = mydict["y_prediksi"], mape = mydict["mape"])

@app.route("/linreg1_spark", methods=["GET", "POST"])
def lr1_spark():
  mydict = dict(linreg1_spark())
  title = 'Linear regression with simulation data using PySpark'
  return render_template('MybigdataApps.html', pname = title, y_aktual = mydict["y_aktual"], y_prediksi = mydict["y_prediksi"], mape = mydict["mape"])

@app.route("/linreg2_spark", methods=["GET", "POST"])
def lr2_spark():
  mydict = dict(linreg2_spark())
  title = 'Linear regression with addjusted simulation data using PySpark'
  return render_template('MybigdataApps.html', pname = title, y_aktual = mydict["y_aktual"], y_prediksi = mydict["y_prediksi"], mape = mydict["mape"])

@app.route("/api/lr1/nonspark", methods=["GET"])
def api1():
  mydict = dict(linreg1_nonspark())
  response = jsonify({'y_aktual': mydict["y_aktual"], 'y_prediksi': mydict["y_prediksi"], 'mape': mydict["mape"]})
  # Enable Access-Control-Allow-Origin
  response.headers.add("Access-Control-Allow-Origin", "*")
  return response

@app.route("/bigdataApps", methods=["GET", "POST"])
def bigdataApps():
  if request.method == 'POST':
    import pandas as pd
    import numpy as np
    import os.path

    #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #url = os.path.join(BASE_DIR, "dataset_dump.csv")

    dataset = request.files['inputDataset']
    # url = "./dataset_dump.csv"

    persentase_data_training = 90
    banyak_fitur = int(request.form['banyakFitur'])
    banyak_hidden_neuron = int(request.form['banyakHiddenNeuron'])
    dataset = pd.read_csv(dataset, delimiter=';', names = ['Tanggal', 'Harga'], usecols=['Harga'])
    # dataset = pd.read_csv(url, delimiter=';', names = ['Tanggal', 'Harga'], usecols=['Harga'])
    # dataset = dataset.fillna(method='ffill')
    minimum = int(dataset.min()-10000)
    maksimum = int(dataset.max()+10000)
    new_banyak_fitur = banyak_fitur + 1
    hasil_fitur = []
    for i in range((len(dataset)-new_banyak_fitur)+1):
      kolom = []
      j = i
      while j < (i+new_banyak_fitur):
        kolom.append(dataset.values[j][0])
        j += 1
      hasil_fitur.append(kolom)
    hasil_fitur = np.array(hasil_fitur)
    data_normalisasi = (hasil_fitur - minimum)/(maksimum - minimum)
    data_training = data_normalisasi[:int(persentase_data_training*len(data_normalisasi)/100)]
    data_testing = data_normalisasi[int(persentase_data_training*len(data_normalisasi)/100):]

    #Training
    bobot = np.random.rand(banyak_hidden_neuron, banyak_fitur)
    bias = np.random.rand(banyak_hidden_neuron)
    h = 1/(1 + np.exp(-(np.dot(data_training[:, :banyak_fitur], np.transpose(bobot)) + bias)))
    h_plus = np.dot(np.linalg.inv(np.dot(np.transpose(h),h)),np.transpose(h))
    output_weight = np.dot(h_plus, data_training[:, banyak_fitur])

    #Testing
    h = 1/(1 + np.exp(-(np.dot(data_testing[:, :banyak_fitur], np.transpose(bobot)) + bias)))
    predict = np.dot(h, output_weight)
    predict = predict * (maksimum - minimum) + minimum

    #MAPE
    aktual = np.array(hasil_fitur[int(persentase_data_training*len(data_normalisasi)/100):, banyak_fitur])
    mape = np.sum(np.abs(((aktual - predict)/aktual)*100))/len(predict)

    print("predict = ", predict)
    print("aktual =", aktual)
    print("mape = ", mape)

    # return render_template('bigdataApps.html', data = {'y_aktual' : list(aktual),'y_prediksi' : list(predict),'mape' : mape})
    return render_template('bigdataApps.html', pname = "Big data apps", y_aktual = list(aktual), y_prediksi = list(predict), mape = mape)


    # return "Big Data Apps " + str(persentase_data_training) + " banyak_fitur = " + str(banyak_fitur) + " banyak_hidden_neuron = " + str(banyak_hidden_neuron) + " :D"
  else:
    return render_template('bigdataApps.html')

def connect_db():
    import os.path

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "fga_big_data_rev2.db")
    # with sqlite3.connect(db_path) as db:

    return sqlite3.connect(db_path)


# cara akses, misal: http://imamcs.pythonanywhere.com/api/fp/3.0/?a=90&b=3&c=2
@app.route("/api/fp/3.0/", methods=["GET"])
# @cross_origin()
def api():
    import os.path
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    url = os.path.join(BASE_DIR, "dataset_dump_tiny.csv")

    # url = "../GGRM.JK.csv"
    # dataset=pd.read_csv(url)

    import pandas as pd
    import numpy as np
    import json
    # from django.http import HttpResponse
    from flask import Response


    a, b, c = request.args.get('a'), request.args.get('b'),request.args.get('c')
    # print(a,' ',b,' ',c)
    # bar = request.args.to_dict()
    # print(bar)

    # dataset = request.FILES['inputDataset']#'E:/Pak Imam/Digitalent/dataset_dump.csv'
    persentase_data_training = int(a)
    banyak_fitur = int(b)
    banyak_hidden_neuron = int(c)
    # print(persentase_data_training,banyak_fitur,banyak_hidden_neuron)

    dataset = pd.read_csv(url, delimiter=';', names = ['Tanggal', 'Harga'], usecols=['Harga'])
    #dataset = pd.read_csv(url, usecols=['Close'])
    dataset = dataset.fillna(method='ffill')

    # print("missing value", dataset.isna().sum())

    minimum = int(dataset.min())
    maksimum = int(dataset.max())
     # print(minimum,maksimum)
    new_banyak_fitur = banyak_fitur + 1
    hasil_fitur = []
    for i in range((len(dataset)-new_banyak_fitur)+1):
        kolom = []
        j = i
        while j < (i+new_banyak_fitur):
            kolom.append(dataset.values[j][0])
            j += 1
        hasil_fitur.append(kolom)
    hasil_fitur = np.array(hasil_fitur)
        # print(hasil_fitur)
    data_normalisasi = (hasil_fitur - minimum)/(maksimum - minimum)

    data_training = data_normalisasi[:int(
        persentase_data_training*len(data_normalisasi)/100)]
    data_testing = data_normalisasi[int(
        persentase_data_training*len(data_normalisasi)/100):]

    # print(data_training)
    # Training
    is_singular_matrix = True
    while(is_singular_matrix):
        bobot = np.random.rand(banyak_hidden_neuron, banyak_fitur)
        #print("bobot", bobot)
        bias = np.random.rand(banyak_hidden_neuron)
        h = 1 / \
            (1 + np.exp(-(np.dot(data_training[:, :banyak_fitur], np.transpose(bobot)) + bias)))

        #print("h", h)
        #print("h_transpose", np.transpose(h))
        #print("transpose dot h", np.dot(np.transpose(h), h))

        # cek matrik singular
        cek_matrik = np.dot(np.transpose(h), h)
        det_cek_matrik = np.linalg.det(cek_matrik)
        if det_cek_matrik != 0:
            #proceed

        #if np.linalg.cond(cek_matrik) < 1/sys.float_info.epsilon:
            # i = np.linalg.inv(cek_matrik)
            is_singular_matrix = False
        else:
            is_singular_matrix = True


    h_plus = np.dot(np.linalg.inv(cek_matrik), np.transpose(h))

    # print("h_plus", h_plus)
    output_weight = np.dot(h_plus, data_training[:, banyak_fitur])

        # print(output_weight)
        # [none,none,...]

    # Testing
    h = 1 / \
        (1 + np.exp(-(np.dot(data_testing[:, :banyak_fitur], np.transpose(bobot)) + bias)))
    predict = np.dot(h, output_weight)
    predict = (predict * (maksimum - minimum) + minimum)

    # MAPE
    aktual = np.array(hasil_fitur[int(
        persentase_data_training*len(data_normalisasi)/100):, banyak_fitur]).tolist()
    mape = np.sum(np.abs(((aktual - predict)/aktual)*100))/len(predict)
    prediksi = predict.tolist()
    # print(prediksi, 'vs', aktual)
    # response = json.dumps({'y_aktual': aktual, 'y_prediksi': prediksi, 'mape': mape})

    # return Response(response, content_type='text/json')
    # return Response(response, content_type='application/json')
    #return Response(response, content_type='text/xml')


    response = jsonify({'y_aktual': aktual, 'y_prediksi': prediksi, 'mape': mape})


    # Enable Access-Control-Allow-Origin
    response.headers.add("Access-Control-Allow-Origin", "*")
    # response.headers.add("access-control-allow-credentials","false")
    # response.headers.add("access-control-allow-methods","GET, POST")


    # r = Response(response, status=200, mimetype="application/json")
    # r.headers["Content-Type"] = "application/json; charset=utf-8"
    return response



# get json data from a url using flask in python
@app.route('/baca_api', methods=["GET"])
def baca_api():
    import requests
    import json
    # uri = "https://api.stackexchange.com/2.0/users?order=desc&sort=reputation&inname=fuchida&site=stackoverflow"
    uri = "http://imamcs.pythonanywhere.com/api/fp/3.0/?a=50&b=3&c=2"
    # try:
    #     uResponse = requests.get(uri)
    # except requests.ConnectionError:
    #     return "Terdapat Error Pada Koneksi Anda"
    # Jresponse = uResponse.text
    # data = json.loads(Jresponse)

    # json.loads(response.get_data().decode("utf-8"))
    data = json.loads(requests.get(uri).decode("utf-8"))
    # data = json.loads(response.get(uri).get_data().decode("utf-8"))

    # import urllib.request
    # with urllib.request.urlopen("http://imamcs.pythonanywhere.com/api/fp/3.0/?a=90&b=3&c=2") as url:
    #     data = json.loads(url.read().decode())
    #     #print(data)

    # from urllib.request import urlopen

    # import json
    # import json
    # store the URL in url as
    # parameter for urlopen
    # url = "https://api.github.com"

    # store the response of URL
    # response = urlopen(url)

    # storing the JSON response
    # from url in data
    # data_json = json.loads(response.read())

    # print the json response
    # print(data_json)

    # data = \
    #     {
    #   "items": [
    #     {
    #       "badge_counts": {
    #         "bronze": 16,
    #         "silver": 4,
    #         "gold": 0
    #       },
    #       "account_id": 258084,
    #       "is_employee": false,
    #       "last_modified_date": 1573684556,
    #       "last_access_date": 1628710576,
    #       "reputation_change_year": 0,
    #       "reputation_change_quarter": 0,
    #       "reputation_change_month": 0,
    #       "reputation_change_week": 0,
    #       "reputation_change_day": 0,
    #       "reputation": 420,
    #       "creation_date": 1292207782,
    #       "user_type": "registered",
    #       "user_id": 540028,
    #       "accept_rate": 100,
    #       "location": "Minneapolis, MN, United States",
    #       "website_url": "http://fuchida.me",
    #       "link": "https://stackoverflow.com/users/540028/fuchida",
    #       "profile_image": "https://i.stack.imgur.com/kP5GW.png?s=128&g=1",
    #       "display_name": "Fuchida"
    #     }
    #   ],
    #   "has_more": false,
    #   "quota_max": 300,
    #   "quota_remaining": 299
    # }

    # displayName = data['items'][0]['display_name']# <-- The display name
    # reputation = data['items'][0]['reputation']# <-- The reputation

    # y_train = data['y_aktual']
    # y_pred = data['y_prediksi']
    # mape = data['mape']

    return data
    # return str(mape)
    # return render_template('MybigdataAppsNonPySpark.html', y_aktual = list(y_train), y_prediksi = list(y_pred), mape = mape)


@app.route('/upload', methods=["GET", "POST"])
def upload():

    form = UploadForm()
    if request.method == "POST":

        if form.validate_on_submit():
            file_name = form.file.data
            database(name=file_name.filename, data=file_name.read() )
            # return render_template("upload.html", form=form)
            return redirect(url_for("dashboard"))

    return render_template("upload.html", form=form)


@app.route('/hapus/file/', methods=["GET"])
def hapus():
    name = request.args.get('name')
    conn = connect_db()
    db = conn.cursor()

    db.execute("DELETE FROM  upload WHERE name =\'"+ name +"\'")
    # mydata
    # for x in c.fetchall():
    #     name_v=x[0]
    #     data_v=x[1]
    #     break

    conn.commit()
    db.close()
    conn.close()

    return redirect(url_for("dashboard"))

@app.route('/unduh/file/', methods=["GET"])
def unduh():
    name = request.args.get('name')
    conn = connect_db()
    db = conn.cursor()

    # c = db.execute(""" SELECT * FROM  upload WHERE name ="""+ name)
    c = db.execute("SELECT * FROM  upload WHERE name =\'"+ name +"\'")
    # mydata
    for x in c.fetchall():
        name_v=x[0]
        data_v=x[1]
        break

    conn.commit()
    db.close()
    conn.close()

    # return render_template('dashboard.html', header = mydata)


    return send_file(BytesIO(data_v), attachment_filename=name_v, as_attachment=True)


@app.route('/download', methods=["GET", "POST"])
def download():

    form = UploadForm()

    if request.method == "POST":
        conn = connect_db()
        db = conn.cursor()

        # conn= sqlite3.connect("fga_big_data_rev2.db")
        # cursor = conn.cursor()
        print("IN DATABASE FUNCTION ")
        c = db.execute(""" SELECT * FROM  upload """)

        for x in c.fetchall():
            name_v=x[0]
            data_v=x[1]
            break

        conn.commit()
        db.close()
        conn.close()

        return send_file(BytesIO(data_v), attachment_filename='flask.pdf', as_attachment=True)


    return render_template("upload.html", form=form)



# class LoginForm(FlaskForm):
class UploadForm(FlaskForm):
    file = FileField()
    submit = SubmitField("submit")
    download = SubmitField("download")

def database(name, data):
    conn = connect_db()
    db = conn.cursor()

    # conn= sqlite3.connect("fga_big_data_rev2.db")
    # cursor = conn.cursor()

    db.execute("""CREATE TABLE IF NOT EXISTS upload (name TEXT,data BLOP) """)
    db.execute("""INSERT INTO upload (name, data) VALUES (?,?) """,(name,data))

    conn.commit()
    db.close()
    conn.close()

def query():
    # conn= sqlite3.connect("fga_big_data_rev2.db")
    # cursor = conn.cursor()

    conn = connect_db()
    db = conn.cursor()

    print("IN DATABASE FUNCTION ")
    c = db.execute(""" SELECT * FROM  upload """)

    for x in c.fetchall():
        name_v=x[0]
        data_v=x[1]
        break



    conn.commit()
    db.close()
    conn.close()

    return send_file(BytesIO(data_v), attachment_filename='flask.pdf', as_attachment=True)

@app.route('/dashboard')
def dashboard():
    # cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    # cur.execute('SELECT * FROM header')
    # data = cur.fetchall()
    # cur.close()

    conn = connect_db()
    db = conn.cursor()

    # conn= sqlite3.connect("fga_big_data_rev2.db")
    # cursor = conn.cursor()
    print("IN DATABASE FUNCTION ")
    c = db.execute(""" SELECT * FROM  upload """)

    mydata = c.fetchall()
    for x in c.fetchall():
        name_v=x[0]
        data_v=x[1]
        break

    hasil = []
    for v_login in c:
        hasil.append(v_login)

    conn.commit()
    db.close()
    conn.close()

    #return send_file(BytesIO(data_v), attachment_filename='flask.pdf', as_attachment=True)

    return render_template('dashboard.html', header = mydata)


@app.route('/logout')
def logout():
   # remove the name from the session if it is there
   session.pop('name', None)
   session.pop('mail', None)
   return redirect(url_for('index'))

@app.route("/emashop", methods=["GET", "POST"])
def gold():
  harga = 919125

  return render_template('emashop.html', gram = harga)


# Free user https://help.pythonanywhere.com/pages/403ForbiddenError/ tidak dapat mengakses API diluar whitelist
@app.route("/prototype", methods=["GET", "POST"])
def prototype():
  import requests

  base_currency = 'IDR'
  symbol = 'XAU'
  endpoint = 'latest'
  access_key = 'ec62hd1le49a170gj5ge4dto47e0del9lw1bi98m85zb3t660695wchlie6n'

  resp = requests.get('https://metals-api.com/api/'+endpoint+'?access_key='+access_key+'&base='+base_currency+'&symbols='+symbol)
  if resp.status_code != 200:
      # This means something went wrong.
      raise ApiError('GET /'+endpoint+'/ {}'.format(resp.status_code))
  api_price = resp.json()
  c24k = api_price['rates']['XAU']/28.35
  import math
  harga = math.ceil(c24k)

  return render_template('emashop.html', gram = harga)

def database_ce(name, data, hasil):
    conn = connect_db()
    db = conn.cursor()

    # conn= sqlite3.connect("fga_big_data_rev2.db")
    # cursor = conn.cursor()

    db.execute("""CREATE TABLE IF NOT EXISTS upload (name TEXT,data BLOP, hasil BLOB) """)
    db.execute("""INSERT INTO upload_ce (name, data, hasil) VALUES (?,?,?) """,(name,data,hasil))

    conn.commit()
    db.close()
    conn.close()

@app.route('/dashboard_ce')
def dashboard_ce():

    conn = connect_db()
    db = conn.cursor()

    # conn= sqlite3.connect("fga_big_data_rev2.db")
    # cursor = conn.cursor()
    print("IN DATABASE FUNCTION ")
    c = db.execute(""" SELECT * FROM  upload_ce """)

    mydata = c.fetchall()
    for x in c.fetchall():
        name_v=x[0]
        data_v=x[1]
        break

    hasil = []
    for v_login in c:
        hasil.append(v_login)

    conn.commit()
    db.close()
    conn.close()

    #return send_file(BytesIO(data_v), attachment_filename='flask.pdf', as_attachment=True)

    return render_template('dashboard_ce.html', header = mydata)

@app.route('/upload_ce', methods=["GET", "POST"])
def upload_ce():

    form = UploadForm()
    if request.method == "POST":

        if form.validate_on_submit():
            file_name = form.file.data
            database_ce(name=file_name.filename, data=file_name.read(), hasil=None )
            # return render_template("upload.html", form=form)
            return redirect(url_for("dashboard_ce"))

    return render_template("upload_ce.html", form=form)

@app.route('/unduh_ce/file/', methods=["GET"])
def unduh_ce():
    name = request.args.get('name')
    conn = connect_db()
    db = conn.cursor()

    c = db.execute("SELECT * FROM  upload_ce WHERE name =\'"+ name +"\'")
    # mydata
    for x in c.fetchall():
        name_v=x[0]
        data_v=x[1]
        break

    conn.commit()
    db.close()
    conn.close()

    # return render_template('dashboard.html', header = mydata)


    return send_file(BytesIO(data_v), attachment_filename=name_v, as_attachment=True)

@app.route('/hapus_ce/file/', methods=["GET"])
def hapus_ce():
    name = request.args.get('name')
    conn = connect_db()
    db = conn.cursor()

    db.execute("DELETE FROM  upload_ce WHERE name =\'"+ name +"\'")

    conn.commit()
    db.close()
    conn.close()

    return redirect(url_for("dashboard_ce"))

def read_ce(name):
    conn = connect_db()
    db = conn.cursor()

    c = db.execute("SELECT * FROM  upload_ce WHERE name =\'"+ name +"\'")
    # mydata
    for x in c.fetchall():
        name_v=x[0]
        data_v=x[1]
        break

    conn.commit()
    db.close()
    conn.close()

    return BytesIO(data_v)

@app.route('/proses_ce/file/', methods=["GET", "POST"])
def proses_ce():
    name = request.args.get('name')

    import numpy as np
    import pandas as pd

    dataset = pd.read_csv(read_ce(name))
    X = dataset['CountAll'].values.reshape(-1,1)
    y = dataset['StockOpen'].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

    # Feature Scaling
    """from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)"""

    # Fitting Simple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    y_pred2 = regressor.predict(X_train)

    aktual, predict = y_train, y_pred2
    mape = np.sum(np.abs(((aktual - predict)/aktual)*100))/len(predict)

    resultdf = pd.DataFrame({'mape': mape,
                   'y_aktual': list(y_train),
                   'y_prediksi': list(y_pred2)})

    nama_hasil = "result_"+name
    hasil_csv = resultdf.to_csv(nama_hasil, encoding='utf-8', index=False)
    hasil_csv = hasil_csv

    conn = connect_db()
    db = conn.cursor()

    db.execute("Update upload_ce set hasil = ? where name = ? ",(hasil_csv,name))

    conn.commit()
    db.close()
    conn.close()

    return redirect(url_for("dashboard_ce"))


@app.route('/get_ce/file/', methods=["GET"])
def get_ce():
    name = request.args.get('name')
    conn = connect_db()
    db = conn.cursor()

    # c = db.execute(""" SELECT * FROM  upload WHERE name ="""+ name)
    c = db.execute("SELECT * FROM  upload_ce WHERE name =\'"+ name +"\'")
    # mydata
    for x in c.fetchall():
        name_v="result_"+x[0]
        data_v=x[2]
        break

    conn.commit()
    db.close()
    conn.close()

    import os.path
    url = os.path.join("/home/unicode12/",name_v)


    return send_file(url, attachment_filename=name_v, as_attachment=True)

if __name__ == '__main__':
    #import os
    #os.environ["JAVA_HOME"] ="/usr/lib/jvm/java-8-openjdk-amd64"
    #print(os.environ["JAVA_HOME"])
    #print(os.environ["SPARK_HOME"])
    #print(os.environ["PYTHONPATH"])
    # db.create_all()
    app.run()  # If address is in use, may need to terminate other sessions:
             # Runtime > Manage Sessions > Terminate Other Sessions
  # app.run(host='0.0.0.0', port=5004)  # If address is in use, may need to terminate other sessions:
             # Runtime > Manage Sessions > Terminate Other Sessions
