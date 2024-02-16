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
# Q.1. Check the count of survived and not survived passengers
survival_count = df.groupBy("Survived").agg(count("PassengerId").alias("Count"))
survival_count.show()

# Q.2. Calculate the average age of passengers
average_age = df.agg(avg("Age").alias("AverageAge")).collect()[0]["AverageAge"]
print(f"Average Age of Passengers: {average_age:.2f}")

# Q.3. Count the number of male and female passengers
gender_count = df.groupBy("Sex").agg(count("PassengerId").alias("Count"))
gender_count.show()

# Q.4. Determine the percentage of passengers in each class
class_percentage = df.groupBy("Pclass").agg((count("PassengerId") / df.count() * 100).alias("Percentage"))
class_percentage.show()

# Q.5.Number of passengers in each class
passengers_by_class= df.groupBy("Pclass").count()
passengers_by_class.show()

# Q.6.Data Preprocessing
# Handling missing values
titanic_df = df.fillna({'Age': average_age, 'Embarked': 'S'})
titanic_df.show()


# Q.7.Convert categorical variables into numerical representations
titanic_df = titanic_df.replace(['male', 'female'], ['0', '1'], 'Sex')
titanic_df = titanic_df.replace(['S', 'C', 'Q'], ['0', '1', '2'], 'Embarked')
titanic_df.show()

# Exploratory Data Analysis (EDA)
# Q.8. Visualize the survival rate by gender
survival_by_gender = df.groupBy("Sex", "Survived").count()
survival_by_gender.show()

# Q.9.Visualize the survival rate by passenger class
survival_by_class = df.groupBy("Pclass", "Survived").count()
survival_by_class.show()

# Q.10.Calculate the maximum age of passengers
max_age = df.agg({"Age": "max"}).collect()[0][0]
print("Maximum age of passengers:", max_age)

# Q.11.Sort passengers by age in descending order
sorted_by_age_desc = df.orderBy(col("Age").desc())
sorted_by_age_desc.show()

# Q.12.Sort passengers by fare in ascending order
sorted_by_fare_asc = df.orderBy(col("Fare").asc())
sorted_by_fare_asc.show()

# Q.13.Calculate the number of siblings/spouses aboard for each passenger
siblings_spouses_count = df.groupBy("SibSp").count()
siblings_spouses_count.show()

# Stop SparkSession
spark.stop()


