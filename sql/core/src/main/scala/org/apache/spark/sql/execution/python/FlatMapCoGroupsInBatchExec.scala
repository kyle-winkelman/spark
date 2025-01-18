/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.execution.python

import org.apache.spark.JobArtifactSet
import org.apache.spark.api.python.ChainedPythonFunctions
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.plans.physical.{AllTuples, ClusteredDistribution, Distribution, Partitioning}
import org.apache.spark.sql.catalyst.types.DataTypeUtils
import org.apache.spark.sql.execution.{CoGroupedIterator, SparkPlan}
import org.apache.spark.sql.execution.python.PandasGroupUtils._

/**
 * Base class for Python-based FlatMapCoGroupsIn*Exec.
 */
trait FlatMapCoGroupsInBatchExec extends SparkPlan with PythonSQLMetrics {
  val groups: Seq[Seq[Attribute]]
  val func: Expression
  val output: Seq[Attribute]
  val children: Seq[SparkPlan]

  protected val pythonEvalType: Int

  private val sessionLocalTimeZone = conf.sessionLocalTimeZone
  private val pythonRunnerConf = ArrowPythonRunner.getPythonRunnerConfMap(conf)
  private val pythonUDF = func.asInstanceOf[PythonUDF]
  private val pandasFunction = pythonUDF.func
  private val chainedFunc =
    Seq((ChainedPythonFunctions(Seq(pandasFunction)), pythonUDF.resultId.id))

  override def producedAttributes: AttributeSet = AttributeSet(output)

  override def outputPartitioning: Partitioning = children.head.outputPartitioning

  override def requiredChildDistribution: Seq[Distribution] =
    groups.map { group =>
      if (group.isEmpty) AllTuples else ClusteredDistribution(group)
    }

  override def requiredChildOrdering: Seq[Seq[SortOrder]] =
    groups.map(_.map(SortOrder(_, Ascending)))

  override protected def doExecute(): RDD[InternalRow] = {
    val (dedups, argOffsets) = children
      .zip(groups)
      .map { case (child, group) =>
        resolveArgOffsets(child.output, group)
      }
      .unzip
    val jobArtifactUUID = JobArtifactSet.getCurrentJobArtifactState.map(_.uuid)

    // Map cogrouped rows to ArrowPythonRunner results, Only execute if partition is not empty
    children.head.execute().zipPartitions(children.tail.map(_.execute()): _*) { iterators =>
      if (iterators.forall(_.isEmpty)) Iterator.empty
      else {
        val groupedIterators = iterators.zip(groups).zip(children).zip(dedups).map {
          case (((iterator, group), child), dedup) =>
            groupAndProject(
              iterator.asInstanceOf[Iterator[InternalRow]],
              group,
              child.output,
              dedup)
        }
        val data = new CoGroupedIterator(groupedIterators, groups.head)
          .map(_._2)

        val runner = new CoGroupedArrowPythonRunner(
          chainedFunc,
          pythonEvalType,
          Array(argOffsets.flatMap(_.iterator).toArray),
          dedups.map(DataTypeUtils.fromAttributes),
          sessionLocalTimeZone,
          pythonRunnerConf,
          pythonMetrics,
          jobArtifactUUID,
          conf.pythonUDFProfiler)

        executePython(data, output, runner)
      }
    }
  }
}
