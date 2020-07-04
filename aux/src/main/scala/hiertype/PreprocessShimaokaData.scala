package hiertype

import scala.collection.JavaConverters._

import poly.io.Local._
import me.tongfei.progressbar._

object PreprocessShimaokaData extends App {

  def normalize(s: String) = s match {
    case "''" => "\""
    case "``" => "\""
    case "-LRB-" => "("
    case "-RRB-" => ")"
    case "-LSB-" => "["
    case "-RSB-" => "]"
    case "-LCB-" => "{"
    case "-RCB-" => "}"
    case _ => s
  }

  val path = args(0)
  for (line <- ProgressBar.wrap(File(path).lines.view.asJava, "Preprocessing").asScala) {
    val Array(strL, strR, strSentence, strTypes, _*) = line.split("\t", 5)
    val l = strL.toInt
    val r = strR.toInt
    val s = strSentence.split(" ").map(normalize)
    val types = strTypes.split(" ").foldLeft(Set[String]()) { (ts, t) =>
      if (ts.exists(t.startsWith)) ts.filterNot(t.startsWith) + t
      else if (ts.exists(_ startsWith t)) ts
      else ts + t
    }

    println(s"${s.mkString(" ").trim}\t${l}:${r}\t${types.mkString(" ")}")
  }

}
