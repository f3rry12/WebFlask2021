def linreg1_spark():
  # MLLIB dari Pyspark Simple Linear Regression /Klasifikasi / Clustering
  # Importing the libraries
  import numpy as np
  import matplotlib.pyplot as plt
  import pandas as pd
  import os

  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  url = os.path.join(BASE_DIR, "Data_Generate_EA.csv")

  import findspark
  findspark.init()

  from pyspark.sql import SparkSession
  spark = SparkSession.builder.appName("Linear Regression Model").getOrCreate()

  from pyspark.ml.regression import LinearRegression
  from pyspark.ml.linalg import Vectors
  from pyspark.ml.feature import VectorAssembler
  from pyspark.ml.feature import IndexToString, StringIndexer

  from pyspark import SQLContext, SparkConf, SparkContext
  from pyspark.sql import SparkSession
  sc = SparkContext.getOrCreate()
  if (sc is None):
      sc = SparkContext(master="local[*]", appName="Linear Regression")
  spark = SparkSession(sparkContext=sc)
  # sqlcontext = SQLContext(sc)

  # Importing the dataset => ganti sesuai dengan case yg anda usulkan
  # a. Min. 30 Data dari case data simulasi dari yg Anda usulkan
  # b. Min. 30 Data dari real case, sesuai dgn yg Anda usulkan dari tugas minggu ke-3 (dari Kaggle/UCI Repository)
  # url = "./Salary_Data.csv"

  sqlcontext = SQLContext(sc)
  data = sqlcontext.read.csv(url, header = True, inferSchema = True)

  from pyspark.ml.feature import VectorAssembler
  # mendifinisikan CountAll sebagai variabel label/predictor
  dataset = data.select(data.CountAll, data.StockOpen.alias('label'))
  # split data menjadi 70% training and 30% testing
  training, test = dataset.randomSplit([0.7, 0.3], seed = 100)
  # mengubah fitur menjadi vektor
  assembler = VectorAssembler().setInputCols(['CountAll',]).setOutputCol('features')
  trainingSet = assembler.transform(training)
  # memilih kolom yang akan di vektorisasi
  trainingSet = trainingSet.select("features","label")

  from pyspark.ml.regression import LinearRegression
  # fit data training ke model
  lr = LinearRegression()
  lr_Model = lr.fit(trainingSet)
  # assembler : fitur menjadi vektor
  testSet = assembler.transform(test)
  # memilih kolom fitur dan label
  testSet = testSet.select("features", "label")
  # fit testing data ke model linear regression
  testSet = lr_Model.transform(testSet)
  # testSet.show(truncate=False)

  from pyspark.ml.evaluation import RegressionEvaluator
  evaluator = RegressionEvaluator()
  # print(evaluator.evaluate(testSet, {evaluator.metricName: "r2"}))

  y_pred2 = testSet.select("prediction")
  # y_pred2.show()

  thisdict =	{
  "y_aktual": y_pred2.rdd.flatMap(lambda x: x).collect(),
  "y_prediksi": y_pred2.rdd.flatMap(lambda x: x).collect(),
  "mape": evaluator.evaluate(testSet, {evaluator.metricName: "r2"})
  }
  return thisdict

def linreg2_spark():
  import numpy as np
  import matplotlib.pyplot as plt
  import pandas as pd
  import os

  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  url = os.path.join(BASE_DIR, "Data_Generated_daily_adjusted_EA.csv")

  import findspark
  findspark.init()

  from pyspark.sql import SparkSession
  spark = SparkSession.builder.appName("Linear Regression Model").getOrCreate()

  from pyspark.ml.regression import LinearRegression
  from pyspark.ml.linalg import Vectors
  from pyspark.ml.feature import VectorAssembler
  from pyspark.ml.feature import IndexToString, StringIndexer

  from pyspark import SQLContext, SparkConf, SparkContext
  from pyspark.sql import SparkSession
  sc = SparkContext.getOrCreate()
  if (sc is None):
      sc = SparkContext(master="local[*]", appName="Linear Regression")
  spark = SparkSession(sparkContext=sc)

  sqlcontext = SQLContext(sc)
  data = sqlcontext.read.csv(url, header = True, inferSchema = True)

  from pyspark.ml.feature import VectorAssembler
  # mendifinisikan CountAll sebagai variabel label/predictor
  dataset = data.select(data.CountAll, data.StockOpen.alias('label'))
  # split data menjadi 70% training and 30% testing
  training, test = dataset.randomSplit([0.7, 0.3], seed = 100)
  # mengubah fitur menjadi vektor
  assembler = VectorAssembler().setInputCols(['CountAll',]).setOutputCol('features')
  trainingSet = assembler.transform(training)
  # memilih kolom yang akan di vektorisasi
  trainingSet = trainingSet.select("features","label")

  from pyspark.ml.regression import LinearRegression
  # fit data training ke model
  lr = LinearRegression()
  lr_Model = lr.fit(trainingSet)
  # assembler : fitur menjadi vektor
  testSet = assembler.transform(test)
  # memilih kolom fitur dan label
  testSet = testSet.select("features", "label")
  # fit testing data ke model linear regression
  testSet = lr_Model.transform(testSet)
  # testSet.show(truncate=False)

  from pyspark.ml.evaluation import RegressionEvaluator
  evaluator = RegressionEvaluator()
  # print(evaluator.evaluate(testSet, {evaluator.metricName: "r2"}))

  y_pred2 = testSet.select("prediction")
  # y_pred2.show()

  thisdict =	{
  "y_aktual": y_pred2.rdd.flatMap(lambda x: x).collect(),
  "y_prediksi": y_pred2.rdd.flatMap(lambda x: x).collect(),
  "mape": evaluator.evaluate(testSet, {evaluator.metricName: "r2"})
  }
  return thisdict