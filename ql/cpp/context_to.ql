/**
 * @name Empty block
 * @kind problem
 * @problem.severity warning
 * @id cpp/example/empty-block
 */

import cpp
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.dataflow.DataFlow
import DataFlow::PathGraph
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.security.Encryption



from VarAccess var, MethodAccess call, VarAccess other, string str, string context
where str = var.getVariable().getName() + var.getVariable().getInitializer().getLocation().toString() and
str in
["passwordfile:///opt/src/src/test/java/com/alibaba/druid/TestRollBack.java:49:27:49:36"]
and
(
    (
        DataFlow::localFlow(DataFlow::exprNode(var), DataFlow::exprNode(other)) and
        context = other.getVariable().getName()
    ) or
    (
        DataFlow::localFlow(DataFlow::exprNode(var), DataFlow::exprNode(call.getAnArgument())) and
        context = call.getMethod().getQualifiedName()
    )

)
select var.getVariable().getName(), var.getVariable().getInitializer().getLocation(), context