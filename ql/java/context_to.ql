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
            var = this.asExpr().(VarAccess).getVariable() and location = var.getInitializer().getLocation().toString() and text = var.getName().toString() |
            (text+location) in
            ["SYSTEM_ACCOUNTfile:///opt/src/src/main/java/io/github/jhipster/sample/config/Constants.java:22:49:22:56", "ERR_INTERNAL_SERVER_ERRORfile:///opt/src/src/main/java/io/github/jhipster/sample/web/rest/errors/ErrorConstants.java:9:60:9:86", "hostfile:///opt/src/src/main/java/io/github/jhipster/sample/config/JHipsterProperties.java:330:35:330:45", "hostfile:///opt/src/src/main/java/io/github/jhipster/sample/config/JHipsterProperties.java:363:35:363:45", "hostfile:///opt/src/src/main/java/io/github/jhipster/sample/config/JHipsterProperties.java:440:35:440:45", "AUTHORIZATION_FAILUREfile:///opt/src/src/main/java/io/github/jhipster/sample/repository/CustomAuditEventRepository.java:25:57:25:79"]
        )
    }
}

class RegularNode extends DataFlow::ExprNode{
    RegularNode(){
        this.asExpr() instanceof VarAccess or
        this.asExpr() instanceof Call
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


// from Variable var, string str
// where str = var.getName() + var.getInitializer().getLocation().toString() and
// str.regexpMatch("SYSTEM_ACCOUNTfile:///opt/src/src/main/java/io/github/jhipster/sample/config/Constants.java:22:49:22:56")
// select var, str

from DataConfig cfg, DataFlow::PathNode source, DataFlow::PathNode sink, string str
where cfg.hasFlowPath(source, sink) and source.getNode() != sink.getNode() and
(
    str = sink.getNode().asExpr().(VarAccess).getVariable().getName() or
    str = sink.getNode().asExpr().(Call).getCallee().toString()
)
select
source.getNode().asExpr().(VarAccess).getVariable().getInitializer().toString(),
source.getNode().asExpr().(VarAccess).getVariable().getInitializer().getLocation().toString(),
str,
sink.getNode().getEnclosingCallable().getName()