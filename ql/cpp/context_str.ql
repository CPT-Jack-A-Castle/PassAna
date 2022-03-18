/**
 * @name Empty block
 * @kind problem
 * @problem.severity warning
 * @id cpp/example/empty-block
 */

import cpp
import semmle.code.cpp.dataflow.DataFlow



from Variable var, FunctionCall call, string str, string context
where str = var.getName() + var.getInitializer().getLocation().toString() and
str in
["privreqfile:///opt/src/nping/NpingOps.cc:2296:22:2296:39"]
and
(
    // (
    //     DataFlow::localFlow(DataFlow::exprNode(var.getInitializer().getExpr()), DataFlow::exprNode(other.getInitializer().getExpr())) and
    //     context = other.getName()
    // ) or
    (
        DataFlow::localFlow(DataFlow::exprNode(var.getInitializer().getExpr()), DataFlow::exprNode(call.getAnArgument())) and
        context = call.getNameQualifier().toString()
    )

)
select var.getName(), var.getInitializer().getLocation(), context


// from FunctionCall call, Variable var
// where
// call.getTarget().getName().toString().regexpMatch("nping.*") and
// call.getAnArgument().getValue() = var.getInitializer().getExpr().getValue()
// select call, var, call.getAnArgument()