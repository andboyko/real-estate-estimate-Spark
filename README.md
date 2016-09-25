# real-estate-estimate-spark

flatPriceEstimate.scala - Raw scala code used for presentation in spark-shell.
Specific data is collected from MySQL, transformed and fitted to model with Spark MLib.

Don't forget to run spark-shell with specific  mysql driver: 
spark-shell --driver-class-path your-path-to-driver/mysql-connector-java-5.1.38/mysql-connector-java-5.1.38-bin.jar

ads_search.scala - sql-like-style read from mysql to Spark DataFrame
