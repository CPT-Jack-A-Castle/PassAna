/**
 * @name Empty block
 * @kind problem
 * @problem.severity warning
 * @id cpp/example/empty-block
 */

import cpp
import semmle.code.cpp.dataflow.TaintTracking
import DataFlow::PathGraph

// predicate isPass(VariableAccess arg){
//    arg.getValueText().regexpMatch("\\w*[Pp]ass\\w*")
// }

// class DataConfig extends TaintTracking::Configuration {
//    DataConfig() { this = "Password" }
//    override predicate isSource(DataFlow::Node nd) {
//       exists(VariableAccess var | isPass(var) and nd.asExpr().(VariableAccess) = var)
//    }
//    override predicate isSink(DataFlow::Node nd) {
//        nd.asExpr() instanceof VariableAccess
//    }
// }

// from DataConfig cfg, DataFlow::PathNode source, DataFlow::PathNode sink
// where cfg.hasFlowPath(source, sink)
// select source, sink

class GetPass extends DataFlow::ExprNode{
   GetPass(){
       exists(VariableAccess var| var = this.asExpr().(VariableAccess) | 
       var.getValueText().regexpMatch("\\w*[Pp]ass\\w*") or
       var.getValueText().regexpMatch("\\w*token\\w*")  or
       var.getValueText().regexpMatch("\\w*Token\\w*")
       )
   }
}

class GetRegularNode extends DataFlow::ExprNode{
   GetRegularNode(){
       exists(VariableAccess var|
            var = this.asExpr().(VariableAccess))
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
source.getNode().asExpr().(VariableAccess).getTarget().getInitializer().getDeclaration(),
sink.getNode().toString() + ";" + 
sink.getNode().asExpr().(VariableAccess).getParent().toString() + ";" +
sink.getNode().getEnclosingCallable().toString()

