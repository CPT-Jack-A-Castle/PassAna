/**
 * @name Empty http
 * @kind problem
 * @problem.severity warning
 * @id java/example/empty-block
 */

import java
import semmle.code.java.dataflow.TaintTracking
import DataFlow::PathGraph
import semmle.code.java.dataflow.DataFlow6
import semmle.code.java.dataflow.FlowSources


class GetString extends DataFlow::ExprNode{
    GetString(){
        exists(Variable var| var = this.asExpr().(VarAccess).getVariable() |
        var.getType() instanceof TypeString
        )
    }
}
 

class DataConfig extends TaintTracking::Configuration {
    DataConfig() {
    this = "..."
  }

  override predicate isSource(DataFlow::Node source) {
    source instanceof RemoteFlowSource
  }

  override predicate isSink(DataFlow::Node sink) {
    sink instanceof GetString
 }
}

from DataConfig cfg, DataFlow::PathNode source, DataFlow::PathNode sink
where cfg.hasFlowPath(source, sink) and source.getNode() != sink.getNode()
select source, sink