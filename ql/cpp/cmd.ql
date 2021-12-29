/**
 * @name Empty block
 * @kind problem
 * @problem.severity warning
 * @id cpp/example/empty-block
 */

import cpp
import semmle.code.cpp.dataflow.TaintTracking
import DataFlow::PathGraph

predicate isPass(VariableAccess arg){
   arg.getValueText().regexpMatch("\\w*[Pp]ass\\w*")
}

class DataConfig extends TaintTracking::Configuration {
   DataConfig() { this = "Password" }
   override predicate isSource(DataFlow::Node nd) {
      exists(VariableAccess var | isPass(var) and nd.asExpr().(VariableAccess) = var)
   }
   override predicate isSink(DataFlow::Node nd) {
       nd.asExpr() instanceof VariableAccess
   }
}

from DataConfig cfg, DataFlow::PathNode source, DataFlow::PathNode sink
where cfg.hasFlowPath(source, sink)
select source, sink