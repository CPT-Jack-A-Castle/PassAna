/**
 * @name Empty block
 * @kind problem
 * @problem.severity warning
 * @id java/example/empty-block
 */

import java
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.dataflow.DataFlow
import DataFlow::PathGraph
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.security.Encryption



from VarAccess var, MethodAccess call, VarAccess other, string str, string context
where str = var.getVariable().getName() + var.getVariable().getInitializer().getLocation().toString() and
str in
["COMPRESSED_FILENAMEfile:///opt/src/src/main/java/com/github/ayltai/gradle/plugin/DownloadTask.java:47:55:47:84"]
and
(
    (
        TaintTracking::localTaint(DataFlow::exprNode(var), DataFlow::exprNode(other)) and
        context = other.getVariable().getName()
    ) or
    (
        TaintTracking::localTaint(DataFlow::exprNode(var), DataFlow::exprNode(call.getAnArgument())) and
        context = call.getMethod().getQualifiedName()
    )

)
select var.getVariable().getName(), var.getVariable().getInitializer().getLocation(), context

// class GetPass extends DataFlow::ExprNode{
//     GetPass(){
//         exists(Variable var, string location, string text|
//             var = this.asExpr().(VarAccess).getVariable() and location = var.getInitializer().getLocation().toString() and text = var.getName().toString() |
//             (text+location) in
//             ["passwordfile:///opt/src/src/test/java/com/alibaba/druid/TestRollBack.java:49:27:49:36"]
//         )
//     }
// }

// class RegularNode extends DataFlow::ExprNode{
//     RegularNode(){
//         this.asExpr() instanceof VarAccess or
//         this.asExpr() instanceof Call
//     }
// }

// class DataConfig extends TaintTracking::Configuration {
//     DataConfig() { this = "<some unique identifier>" }
//     override predicate isSource(DataFlow::Node nd) {
//        nd instanceof GetPass
//     }
//     override predicate isSink(DataFlow::Node nd) {
//         nd.asExpr() instanceof Assignment
//     }

// }
