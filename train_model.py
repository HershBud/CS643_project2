# train_model.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import when, col
from pyspark import SparkFiles 

# Configuration based on your m5.large workers (8 GiB RAM, 2 cores)
spark = SparkSession.builder \
    .appName("WineQualityPredictionTraining") \
    .master("spark://master:7077") \
    .config("spark.executor.memory", "6g") \
    .config("spark.executor.cores", "2") \
    .config("spark.cores.max", "6") \
    .getOrCreate()

# ----------------- CONFIGURATION & DISTRIBUTION -----------------

# The file name and its full path on the Master Node's local disk
FILE_NAME = "TrainingDataset_noheader.csv"
# We use the full 'file:///' path of the Master node to ensure addFile finds it
FILE_PATH_ON_MASTER = f"file:///home/ubuntu/CS643_project2/{FILE_NAME}"

# Step 1: Programmatically send the data file to all worker nodes
print(f"Distributing training data file ({FILE_NAME}) to all worker nodes...")
spark.sparkContext.addFile(FILE_PATH_ON_MASTER)

# Column names must match the official dataset structure
column_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality"
]

# ----------------- DATA LOADING -----------------

# Step 2: Workers access the file from their local cache directory
data = spark.read.csv(
    SparkFiles.get(FILE_NAME), # <--- Accesses the locally cached file on the worker
    sep=";", 
    inferSchema=True, 
    header=False  
)

data = data.toDF(*column_names)

# --------------- DATA PREPARATION (Unchanged) ---------------

# 1. Feature Engineering: Assemble all features into a single vector
feature_columns = column_names[:-1]
assembler = VectorAssembler(
    inputCols=feature_columns, 
    outputCol="features"
)
data = assembler.transform(data)

# 2. Target Variable: Convert the 'quality' score into a binary class.
data = data.withColumn(
    "label", 
    when(col("quality") >= 6, 1.0).otherwise(0.0)
)

# Select final columns and split
final_data = data.select("label", "features")
(training_data, testing_data) = final_data.randomSplit([0.7, 0.3], seed=42)

# --------------- PARALLEL MODEL TRAINING ---------------

print("\nStarting parallel Random Forest training on the Spark cluster...")
rf = RandomForestClassifier(
    labelCol="label", 
    featuresCol="features", 
    numTrees=100, 
    maxDepth=10, 
    seed=42
)

# Train the model (This step is distributed across the 3 Worker Nodes)
model = rf.fit(training_data)
print("Model training complete.")

# --------------- EVALUATION AND SAVING ---------------

predictions = model.transform(testing_data)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)

print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")

# Save the trained model to the Master Node's local disk
MODEL_PATH = "file:///home/ubuntu/CS643-project2/wine_quality_model"
try:
    model.write().overwrite().save(MODEL_PATH)
    print(f"\nTrained model successfully saved to: {MODEL_PATH}")
except Exception as e:
    print(f"\nERROR saving model: {e}")

spark.stop()