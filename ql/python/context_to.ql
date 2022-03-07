/**
 * @name Empty block
 * @kind problem
 * @problem.severity warning
 * @id java/example/empty-block
 */

import python
import semmle.python.dataflow.new.TaintTracking
import semmle.python.dataflow.new.DataFlow

class GetPass extends DataFlow::ExprNode{
    GetPass(){
        exists(Unicode uni, string location, string text |uni = this.asExpr()  and location = uni.getLocation().toString() and text = uni.getText() | (text+location) in
        ["/etc/hosts/opt/src/tripleo_ansible/tests/plugins/filter/test_helpers.py:380"]
        )
    }
}


//	/opt/src/tripleo_ansible/tests/plugins/filter/test_helpers.py:380
class RegularNode extends DataFlow::ExprNode{
    RegularNode(){
        exists(Call call, Name name|
            this.asExpr().(Name)= name or this.asExpr().(Call) = call)

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

from DataConfig cfg, DataFlow::PathNode source, DataFlow::PathNode sink, string str
where cfg.hasFlowPath(source, sink) and (str = sink.getNode().asExpr().(Name).getId() or str = sink.getNode().asExpr().(Call).getFunc().getASubExpression().toString())
select
source.getNode().asExpr().(Unicode).getText(), source.getNode().getLocation(),
str, sink.getNode().getScope().getName()

