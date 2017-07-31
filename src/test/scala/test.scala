import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics

object test {
  def main(args: Array[String]): Unit = {
    val testData: Array[Double] = Array(
      1889, 1651, 1561, 1778,
      2403, 2048, 2087, 2197,
      2119, 1700, 1815, 2222,
      1645, 1627, 1110, 1533,
      1976, 1916, 1614, 1883,
      1712, 1712, 1439, 1546,
      1943, 1685, 1271, 1671,
      2104, 1820, 1717, 1874,
      2983, 2794, 2412, 2581,
      1745, 1600, 1384, 1508,
      1710, 1591, 1518, 1667,
      2046, 1907, 1627, 1898,
      1840, 1841, 1595, 1741,
      1867, 1685, 1493, 1678,
      1859, 1649, 1389, 1714,
      1954, 2149, 1180, 1281,
      1325, 1170, 1002, 1176,
      1419, 1371, 1252, 1308,
      1828, 1634, 1602, 1755,
      1725, 1594, 1313, 1646,
      2276, 2189, 1547, 2111,
      1899, 1614, 1422, 1477,
      1633, 1513, 1290, 1516,
      2061, 1867, 1646, 2037,
      1856, 1493, 1356, 1533,
      1727, 1412, 1238, 1469,
      2168, 1896, 1701, 1834,
      1655, 1675, 1414, 1597,
      2326, 2301, 2065, 2234,
      1490, 1382, 1214, 1284
    )
    val testV = new org.apache.spark.ml.linalg.DenseVector(Array(1889.0, 1651.0, 1561, 1778))
    val testDM = new org.apache.spark.ml.linalg.DenseMatrix(4, 30, testData)
    val testDM2 = testDM.transpose
    val myLoci = new LOCI("euclidean")


  //  val res = myLoci.LOCI(testV, testDM2, 0.5)
  //  println(res)
    val res=myLoci.LOCI(testDM2,0.5)
    res.map(println)
  }
}
