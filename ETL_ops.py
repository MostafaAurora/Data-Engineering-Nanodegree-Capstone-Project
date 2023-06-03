# Import Libraries
from numpy import mean
from pyspark.sql.functions import  col, udf
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import *
from pyspark.sql import functions as F
import helpers as helpers





# Define functions to perform operations in the jupyter notebook

def create_migrant_dimension(input_df, output_data):
    """
    Transforms migrant data and writes it to parquet files.

    Description: 
        This function takes an input DataFrame containing migrant data, transforms it by adding a `migrant_id` column,
        selecting relevant columns, renaming the `biryear` column to `birth_year`, and removing duplicate rows. The
        transformed data is then written to parquet files at the specified output path.

    Parameters:
        input_df: A DataFrame containing migrant data.
        output_data: The path where the output parquet files should be written.

    Returns:
        migrants_df: A DataFrame representing the transformed migrant data.
    """

    migrants_df = input_df.withColumn("migrant_id", monotonically_increasing_id()) \
                .select(["migrant_id", "biryear", "gender"]) \
                .withColumnRenamed("biryear", "birth_year")\
                .dropDuplicates(["birth_year", "gender"])
    
    helpers.write_to_parquet(migrants_df, output_data, "migrant")
    return migrants_df




def create_status_dimension(input_df, output_data):
    """
    Transforms status data and writes it to parquet files.

    Description: 
        This function takes an input DataFrame containing status data, transforms it by adding a `status_flag_id` column,
        selecting relevant columns, renaming several columns, and removing duplicate rows. The transformed data is then
        written to parquet files at the specified output path.

    Parameters:
        input_df: A DataFrame containing status data.
        output_data: The path where the output parquet files should be written.

    Returns:  
        status_df: A DataFrame representing the transformed status data.
    """

    status_df = input_df.withColumn("status_flag_id", monotonically_increasing_id()) \
                .select(["status_flag_id", "entdepa", "entdepd", "matflag"]) \
                .withColumnRenamed("entdepa", "arrival_flag")\
                .withColumnRenamed("entdepd", "departure_flag")\
                .withColumnRenamed("matflag", "match_flag")\
                .dropDuplicates(["arrival_flag", "departure_flag", "match_flag"])
    
    helpers.write_to_parquet(status_df, output_data, "status")
    return status_df




def create_visa_dimension(input_df, output_data):
    """
    Transforms visa data and writes it to parquet files.

    Description:
        This function takes an input DataFrame containing visa data, transforms it by adding a `visa_id` column,
        selecting relevant columns, and removing duplicate rows. The transformed data is then written to parquet files
        at the specified output path.

    Parameters:
        :param input_df: A DataFrame containing visa data.
        :param output_data: The path where the output parquet files should be written.

    Returns:
        visa_df: A DataFrame representing the transformed visa data.
    """
    
    visa_df = input_df.withColumn("visa_id", monotonically_increasing_id()) \
                .select(["visa_id","i94visa", "visatype", "visapost"]) \
                .dropDuplicates(["i94visa", "visatype", "visapost"])
    
    helpers.write_to_parquet(visa_df, output_data, "visa")
    return visa_df




def create_state_dimension(input_df, output_data):
    """
    Transforms state data and writes it to parquet files.

    Description:
        This function takes an input DataFrame containing state data, transforms it by selecting relevant columns,
        renaming them to more descriptive names, grouping the data by state code and state name, and calculating
        several aggregate measures such as median age, total population, male population, female population,
        foreign-born population, and average household size. The transformed data is then written to parquet files
        at the specified output path.

    Parameters:
        input_df: A DataFrame containing state data.
        output_data: The path where the output parquet files should be written.

    Returns:
        state_df: A DataFrame representing the transformed state data.
    """

    state_df = input_df.select(["State Code", "State", "Median Age", "Male Population", "Female Population", "Total Population", "Average Household Size",\
                          "Foreign-born", "Race", "Count"])\
                .withColumnRenamed("State Code", "state_code")\
                .withColumnRenamed("Median Age", "median_age")\
                .withColumnRenamed("Male Population", "male_population")\
                .withColumnRenamed("Female Population", "female_population")\
                .withColumnRenamed("Total Population", "total_population")\
                .withColumnRenamed("Average Household Size", "average_household_size")\
                .withColumnRenamed("Foreign-born", "foreign_born")
    
    state_df = state_df.groupBy(col("state_code"), col("State").alias("state")).agg(
                round(mean('median_age'), 2).alias("median_age"),\
                sum("total_population").alias("total_population"),\
                sum("male_population").alias("male_population"), \
                sum("female_population").alias("female_population"),\
                sum("foreign_born").alias("foreign_born"), \
                round(mean("average_household_size"),2).alias("average_household_size")
                ).dropna()
    
    helpers.write_to_parquet(state_df, output_data, "state")
    return state_df




def create_time_dimension(input_df, output_data):
    """
    Transforms time data and writes it to parquet files.

    Description:
        This function takes an input DataFrame containing time data in SAS format, transforms it by converting the
        SAS date to a datetime object and extracting several time-related features such as day, month, year, week,
        and weekday. The transformed data is then written to parquet files at the specified output path.

    Parameters:
        input_df: A DataFrame containing time data in SAS format.
        output_data: The path where the output parquet files should be written.

    Returns:
        time_df: A DataFrame representing the transformed time data.
    """
    from datetime import datetime, timedelta
    from pyspark.sql import types as T
    
    def convert_datetime(x):
        try:
            start = datetime(1960, 1, 1)
            return start + timedelta(days=int(x))
        except:
            return None
    
    udf_datetime_from_sas = udf(lambda x: convert_datetime(x), T.DateType())

    time_df = input_df.select(["arrdate"])\
                .withColumn("arrival_date", udf_datetime_from_sas("arrdate")) \
                .withColumn('day', F.dayofmonth('arrival_date')) \
                .withColumn('month', F.month('arrival_date')) \
                .withColumn('year', F.year('arrival_date')) \
                .withColumn('week', F.weekofyear('arrival_date')) \
                .withColumn('weekday', F.dayofweek('arrival_date'))\
                .select(["arrdate", "arrival_date", "day", "month", "year", "week", "weekday"])\
                .dropDuplicates(["arrdate"])
    
    helpers.write_to_parquet(time_df, output_data, "time")
    return time_df




def create_airport_dimension(input_df, output_data):
    """
    Transforms airport data and writes it to parquet files.

    Description:
        This function takes an input DataFrame containing airport data, transforms it by selecting relevant columns
        and removing duplicate rows based on the `ident` column. The transformed data is then written to parquet files
        at the specified output path.

    Parameters:
        input_df: A DataFrame containing airport data.
        output_data: The path where the output parquet files should be written.

    Returns:
        airport_df: A DataFrame representing the transformed airport data.
    """
    
    airport_df = input_df.select(["ident", "type", "iata_code", "name", "iso_country", "iso_region", "municipality", "gps_code", "coordinates", "elevation_ft"])\
                .dropDuplicates(["ident"])
    
    helpers.write_to_parquet(airport_df, output_data, "airport")
    return airport_df




def create_temperature_dimension(input_df, output_data):
    """
    Transforms temperature data and writes it to parquet files.

    Description:
        This function takes an input DataFrame containing temperature data, transforms it by grouping the data by
        country and calculating the mean average temperature and average temperature uncertainty. The transformed
        data is then written to parquet files at the specified output path.

    Parameters:
        input_df: A DataFrame containing temperature data.
        output_data: The path where the output parquet files should be written.

    Returns:
        temperature_df: A DataFrame representing the transformed temperature data.
    """

    temperature_df = input_df.groupBy(col("Country").alias("country")).agg(
                round(mean('AverageTemperature'), 2).alias("average_temperature"),\
                round(mean("AverageTemperatureUncertainty"),2).alias("average_temperature_uncertainty")
            ).dropna()\
            .withColumn("temperature_id", monotonically_increasing_id()) \
            .select(["temperature_id", "country", "average_temperature", "average_temperature_uncertainty"])
    
    helpers.write_to_parquet(temperature_df, output_data, "temperature")
    return temperature_df




def create_country_dimension(input_df, output_data):
    """
    Writes country data to parquet files.

    Description:
        This function takes an input DataFrame containing country data and writes it to parquet files at the specified
        output path. No transformations are performed on the input data.

    Parameters:
        input_df: A DataFrame containing country data.
        output_data: The path where the output parquet files should be written.

    Returns:
        country_df: A DataFrame representing the country data.
    """

    country_df = input_df
    helpers.write_to_parquet(country_df, output_data, "country")
    return country_df




def create_immigration_fact(immigration_spark, output_data, spark):
    """
    Transforms immigration data and writes it to parquet files.

    Description:
        This function takes an input DataFrame containing immigration data and several other DataFrames containing
        related data such as airport, country temperature, migrant, state, status, time, and visa. The function
        performs several join operations to combine the data from these DataFrames with the immigration data. The
        resulting data is then written to parquet files at the specified output path.

    Parameters:
        immigration_spark: A DataFrame containing immigration data.
        output_data: The path where the output parquet files should be written.
        spark: A SparkSession object.

    Returns:
        immigration_df: A DataFrame representing the transformed immigration data.
    """

    airport = spark.read.parquet("tables/airport")
    country_temperature = spark.read.parquet("tables/country_temperature_mapping")
    migrant = spark.read.parquet("tables/migrant")
    state = spark.read.parquet("tables/state")
    status = spark.read.parquet("tables/status")
    time = spark.read.parquet("tables/time")
    visa = spark.read.parquet("tables/visa")

    # join all tables to immigration
    immigration_df = immigration_spark.select(["*"])\
                .join(airport, (immigration_spark.i94port == airport.ident), how='full')\
                .join(country_temperature, (immigration_spark.i94res == country_temperature.country_code), how='full')\
                .join(migrant, (immigration_spark.biryear == migrant.birth_year) & (immigration_spark.gender == migrant.gender), how='full')\
                .join(status, (immigration_spark.entdepa == status.arrival_flag) & (immigration_spark.entdepd == status.departure_flag) &\
                      (immigration_spark.matflag == status.match_flag), how='full')\
                .join(visa, (immigration_spark.i94visa == visa.i94visa) & (immigration_spark.visatype == visa.visatype)\
                      & (immigration_spark.visapost == visa.visapost), how='full')\
                .join(state, (immigration_spark.i94addr == state.state_code), how='full')\
                .join(time, (immigration_spark.arrdate == time.arrdate), how='full')\
                .where(col('cicid').isNotNull())\
                .select(["cicid", "i94res", "depdate", "i94mode", "i94port", "i94cit", "i94addr", "airline", "fltno", "ident", "country_code",\
                         "temperature_id", "migrant_id", "status_flag_id", "visa_id", "state_code", time.arrdate.alias("arrdate")])
    
    helpers.write_to_parquet(immigration_df, output_data, "immigration")
    return immigration_df