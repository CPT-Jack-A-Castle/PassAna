/**
 * @name Empty block
 * @kind problem
 * @problem.severity warning
 * @id java/example/empty-block
 */

import java
import semmle.code.java.dataflow.TaintTracking
import DataFlow::PathGraph
import semmle.code.java.dataflow.FlowSources



class GetPassSink extends DataFlow::ExprNode{
    GetPassSink(){
        exists(Variable var| var = this.asExpr().(VarAccess).getVariable() | var.getName().regexpMatch("\\w*[Pp]ass\\w*"))
    }
}

class GetPassSource extends DataFlow::ExprNode{
    GetPassSource(){
        exists(Variable var|
             var = this.asExpr().(VarAccess).getVariable())
    }
}

class DataConfig extends TaintTracking::Configuration {
    DataConfig() { this = "<some unique identifier>" }
    override predicate isSource(DataFlow::Node nd) {
       nd instanceof GetPassSource
    }
    override predicate isSink(DataFlow::Node nd) {
        nd instanceof GetPassSink}
}

from DataConfig cfg, DataFlow::PathNode source, DataFlow::PathNode sink
where cfg.hasFlowPath(source, sink)
select
sink.getNode().asExpr().getControlFlowNode().toString(),
source.getNode().asExpr().(VarAccess).getVariable().getInitializer().toString() + "-" +
source.getNode().asExpr().(VarAccess).getVariable().getInitializer().getLocation().getStartLine(),
source.getNode().asExpr().(VarAccess).getVariable().getName().toString(),
source.getNode().asExpr().(VarAccess).getVariable().getType().toString()