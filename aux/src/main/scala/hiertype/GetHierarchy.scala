package hiertype

import scala.collection._
import scala.collection.JavaConverters._

import me.tongfei.progressbar._
import poly.io.Local._

object GetHierarchy extends App {

  val types = mutable.HashSet[String]()

  val path = args(0)
  for (line <- ProgressBar.wrap(File(path).lines.view.asJava, "Processing").asScala) {
    val Array(_, _, strTypes) = line.split("\t", 3)
    val ts = strTypes.split(" ")
    ts foreach types.add
  }

  types.toArray.sorted foreach println

}
