val ads_conf = sqlContext.read.jdbc(url,"ad",Array("c_content like '%конферен%'"),prop)
val conf_prepared = ads_conf.select("id","c_content","phone","email")
conf_prepared.show