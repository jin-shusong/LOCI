/* @(#)LOCI.scala
 */
/**
  *
  * Time-stamp: <2017-07-22 12:24:33 jinss>
  * Author: jinss
  *
  * @author <a href="mailto:(shusong.jin@Istuary.com)">Jin Shusong</a>
  *         Version: $Id: LOCI.scala,v 0.0 2017/07/21 03:08:59 jinss Exp$
  *         \revision$Header: /home/jinss/study/IdeaProjects/LOCI/src/main/scala/LOCI.scala,v 0.0 2017/07/21 03:08:59 jinss Exp$
  */

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics
import org.apache.spark.ml.linalg.{DenseVector, DenseMatrix}

class LOCI(val distanceMeasure: String) extends myUtil {

  def LOCI(y: DenseMatrix, alpha: Double): Array[Boolean] = {
    require(alpha > 0.0 && alpha < 1.0)
    val yIter = y.rowIter
    yIter.map(row => LOCI(row.toDense, y, alpha)).toArray

  }

  def LOCI(x: DenseVector, y: DenseMatrix,
           alpha: Double): Boolean = {
    require(alpha > 0.0 && alpha < 1.0)
    val distVec = distanceVector(x, y, distanceMeasure)
    val distMatrix = distanceMatrix(y, distanceMeasure)
    val maxDis1 = distMatrix.toArray.max

    val maxDis2 = distVec(distVec.argmax)
    val maxDis = if (maxDis1 > maxDis2) maxDis1 else maxDis2
    val rp = maxDis / alpha
    val tmpDiff = (rp / 16.0 - rp / 25.0) / 9.0
    val rpSeq = (0 until 10).map(i => rp / 25.0 + i * tmpDiff)
    val rpArray = rpSeq.toArray
    val resTmp: Array[(Double, Double)] = rpArray.map(delta => MDEF(x, y, 0.5 * delta, delta))
    val resMDEF = resTmp.map(res => res._1)
    val resSigma = resTmp.map(res => res._2)
    val resBoolean = (0 until 10).map(i => resMDEF(i) >= 3.0 * resSigma(i))
    val res = resBoolean.filter(_ == true)
    if (res.length > 9) true else false
  }

  def averageMandStd(x: DenseVector,
                     y: DenseMatrix,
                     epsilon: Double,
                     delta: Double): (Double, Double) = {
    require(epsilon < delta && epsilon > 0.0)
    val tmpTuple = insideXpointEpsilonNeighbor(x, y, delta)
    val xNeigh = tmpTuple._1
    val tmpSeq = xNeigh.map { i =>
      val yTmpVec = oneRowOfSparkDenseMatrix(i, y)
      val dVwith = distanceVector(yTmpVec, y, distanceMeasure)
      val choiceTmp = dVwith.values.count(_ <= epsilon)
      choiceTmp
    }
    val stats = new DescriptiveStatistics()
    for (i <- tmpSeq) stats.addValue(i.toDouble)
    val tmpLeng = tmpSeq.length
    val mean = if (tmpLeng == 1) tmpSeq(0) else stats.getMean
    val std = if (tmpLeng == 1) 0.0 else stats.getStandardDeviation
    (mean, std)
  }

  def MDEF(x: DenseVector,
           y: DenseMatrix,
           epsilon: Double,
           delta: Double): (Double, Double) = {
    require(delta > epsilon && epsilon > 0.0)
    val tmp1 = functionM(x, y, epsilon)
    val tmp2 = averageMandStd(x, y, epsilon, delta)
    val tmp2Mean = tmp2._1
    val tmp2Std = tmp2._2
    val mdef = 1.0 - tmp1 / tmp2Mean
    val sigma = tmp2Std / tmp2Mean
    (mdef, sigma)
  }

  def functionM(x: DenseVector,
                y: DenseMatrix,
                epsilon: Double): Int = {
    val tmp = insideXpointEpsilonNeighbor(x, y, epsilon)
    tmp._1.length
  }

  def insideXpointEpsilonNeighbor(x: DenseVector,
                                  y: DenseMatrix,
                                  epsilon: Double): (Array[Int], Array[Double]) = {
    val distVec = distanceVector(x, y, distanceMeasure)
    val nrow = y.numRows
    val choiceTmp = (0 until nrow).filter(distVec(_) <= epsilon)
    val choiceVec = choiceTmp.toArray
    val distVec2 = choiceVec.map(i => distVec(i))
    (choiceVec, distVec2)
  }

}
