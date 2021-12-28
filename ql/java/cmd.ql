/**
 * @name Empty block
 * @kind problem
 * @problem.severity warning
 * @id java/example/empty-block
 */

import java
import semmle.code.java.dataflow.TaintTracking
import DataFlow::PathGraph

predicate isPass(Variable arg){
    arg.getName().regexpMatch("\\w*[Pp]ass\\w*")
}

class DataConfig extends TaintTracking::Configuration {
    DataConfig() { this = "<some unique identifier>" }
    override predicate isSource(DataFlow::Node nd) {
       exists(Variable var | isPass(var) and nd.asExpr().(VarAccess).getVariable() = var)
    }
    override predicate isSink(DataFlow::Node nd) {
        nd.asExpr() instanceof VarAccess
    }
}

// from DataConfig cfg, DataFlow::PathNode source, DataFlow::PathNode sink
// where cfg.hasFlowPath(source, sink)
// select source.toString(), sink.toString()

from BlockStmt b
where b.getNumStmt() = 0
select b, "This is an empty block."

