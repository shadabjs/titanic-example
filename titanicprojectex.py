from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, avg

# Create a Spark session
spark = SparkSession.builder.appName("TitanicProject").getOrCreate()

# Load the Titanic CSV file into a DataFrame
csv_file_path = "titanic (1) (1).csv"
df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Show the schema and first few rows of the DataFrame
df.printSchema()
df.show(5)

# Perform basic EDA
# 1. Check the count of survived and not survived passengers
survival_count = df.groupBy("Survived").agg(count("PassengerId").alias("Count"))
survival_count.show()

# 2. Calculate the average age of passengers
average_age = df.agg(avg("Age").alias("AverageAge")).collect()[0]["AverageAge"]
print(f"Average Age of Passengers: {average_age:.2f}")

# 3. Count the number of male and female passengers
gender_count = df.groupBy("Sex").agg(count("PassengerId").alias("Count"))
gender_count.show()

# 4. Determine the percentage of passengers in each class
class_percentage = df.groupBy("Pclass").agg((count("PassengerId") / df.count() * 100).alias("Percentage"))
class_percentage.show()

# Stop the Spark session
spark.stop()