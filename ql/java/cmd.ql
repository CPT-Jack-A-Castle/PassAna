/**
 * @name Empty block
 * @kind problem
 * @problem.severity warning
 * @id java/example/empty-block
 */

import java
import semmle.code.java.dataflow.TaintTracking
import DataFlow::PathGraph
import semmle.code.java.dataflow.DataFlow6
import semmle.code.java.security.Encryption

class GetPass extends DataFlow::ExprNode{
    GetPass(){
        exists(Variable var| var = this.asExpr().(VarAccess).getVariable()|
        var.getName().regexpMatch("\\w*[Pp]ass\\w*") or
        var.getName().regexpMatch("\\w*token\\w*")  or
        var.getName().regexpMatch("\\w*Token\\w*")
        )
    }
}

class GetRegularNode extends DataFlow::ExprNode{
    GetRegularNode(){
        exists(Variable var|
             var = this.asExpr().(VarAccess).getVariable())
    }
}

class DataConfig extends TaintTracking::Configuration {
    DataConfig() { this = "<some unique identifier>" }
    override predicate isSource(DataFlow::Node nd) {
       nd instanceof GetPass
    }
    override predicate isSink(DataFlow::Node nd) {
        nd instanceof GetRegularNode}

}



from DataConfig cfg, DataFlow::PathNode source, DataFlow::PathNode sink
where cfg.hasFlowPath(source, sink) and source.getNode() != sink.getNode()
select
source.toString(),
source.getNode().asExpr().(VarAccess).getVariable().getInitializer().toString() + source.getNode().asExpr().(VarAccess).getVariable().getInitializer().getLocation().getStartLine(),
sink.getNode().toString() + ";" +
sink.getNode().asExpr().(VarAccess).getParent().toString() + ";" +
sink.getNode().getEnclosingCallable().toString()