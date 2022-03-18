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

