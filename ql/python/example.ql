/**
 * @name python scope
 * @kind problem
 * @problem.severity warning
 * @id python/example/empty-scope
 */

import python
private import semmle.python.dataflow.new.DataFlow
import DataFlow::PathGraph
import semmle.python.security.dataflow.CleartextLogging::CleartextLogging

from Configuration config, DataFlow::PathNode source, DataFlow::PathNode sink, string classification
where
  config.hasFlowPath(source, sink) and
  classification = source.getNode().(Source).getClassification()
select sink.getNode(), source, sink, "$@ is logged here.", source.getNode(),
  "Sensitive data (" + classification + ")"