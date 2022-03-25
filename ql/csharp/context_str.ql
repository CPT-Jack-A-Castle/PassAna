/**
 * @name Empty block
 * @kind problem
 * @problem.severity warning
 * @id csharp/example/empty-block
 */

import csharp


from VariableAccess var, VariableAccess other, Call call,  string str, string context
where str = var.getTarget().getName() + var.getTarget().getInitializer().getLocation().toString() and
str in
["_sConnect/opt/src/WsBenchmark/Controllers/InterController.cs:12:24:12:32"]
and
(
    (
        TaintTracking::localTaint(DataFlow::exprNode(var), DataFlow::exprNode(other)) and
        context = other.getTarget().getName()
    ) or
    (
        DataFlow::localFlow(DataFlow::exprNode(var), DataFlow::exprNode(call.getAnArgument())) and
        context = call.toString()
    )

)
select var.getTarget().getName(), var.getTarget().getInitializer().getLocation(), context