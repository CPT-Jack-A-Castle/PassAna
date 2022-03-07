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
        exists(Variable var, string location, string text|
            var = this.asExpr().(VarAccess).getVariable() and location = var.getLocation().toString() and text = var.getName().toString() |
            (text+location) in
            ["dcatfile:///opt/src/dcatap/src/main/java/org/n52/helgoland/adapters/dcat/CatalogTransformer.java:118:33:118:43"]
        )
    }
}

class RegularNode extends DataFlow::ExprNode{
    RegularNode(){
        this.asExpr() instanceof VarAccess
    }
}

class DataConfig extends TaintTracking::Configuration {
    DataConfig() { this = "<some unique identifier>" }
    override predicate isSource(DataFlow::Node nd) {
       nd instanceof GetPass
    }
    override predicate isSink(DataFlow::Node nd) {
        nd instanceof RegularNode
    }

}

// from DataConfig cfg, DataFlow::PathNode source, DataFlow::PathNode sink
// where cfg.hasFlowPath(source, sink)
// select source, sink
from DataConfig cfg, DataFlow::PathNode source, DataFlow::PathNode sink, string str
where cfg.hasFlowPath(source, sink) and source.getNode() != sink.getNode() and str = sink.getNode().asExpr().(StringLiteral).getValue()
select
source.toString(),
source.getNode().asExpr().(VarAccess).getVariable().getInitializer().getLocation(),
str, sink.getNode().asExpr().getEnclosingCallable().getName()