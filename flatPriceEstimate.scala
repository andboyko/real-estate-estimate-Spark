// set properties and read sql
val url = "jdbc:mysql://localhost:3306/dom"
val prop = new java.util.Properties
prop.setProperty("user","root")
prop.setProperty("password","")
prop.setProperty("useSSL","false")
val kv = sqlContext.read.jdbc(url,"ad_flat_sell",prop)

// 1 room, "id_area" =  15 - Салтовcкое (С-В) напр.
val salt_0815_1 = sqlContext.read.jdbc(url,"ad_flat_sell", Array("added > 2015-08-29 and rooms = 1 and id_area = 15 and price BETWEEN 12000 and 30000"), prop)

// 
val salt_1 = salt_0815_1.select("price", "floors", "area_general").filter("area_general is not null")

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
val data = salt_1.map { r =>  LabeledPoint(new java.lang.Integer(r.getInt(0)).doubleValue(), Vectors.dense(r.getInt(1), r.getDouble(2))) }
data.count

val splits=data.randomSplit(Array (0.8,0.2));
val training=splits(0).cache;
val test=splits(1).cache;
training.first
test.first

import org.apache.spark.ml.regression.LinearRegression
import sqlContext.implicits._
val tr_df = training.toDF()
val lr = new LinearRegression()
lr.setMaxIter(10)
lr.setRegParam(0.3)
lr.setElasticNetParam(0.8)
val lrModel = lr.fit(tr_df)
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

val trainingSummary = lrModel.summary
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")

val est1 = lrModel.coefficients.apply(0)*test.first.features.apply(0) + lrModel.coefficients.apply(1) * test.first.features.apply(1) + lrModel.intercept
val est2 = lrModel.coefficients.apply(0)*training.first.features.apply(0) + lrModel.coefficients.apply(1) * training.first.features.apply(1) + lrModel.intercept
