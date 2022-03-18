/**
 * @name Empty block
 * @kind problem
 * @problem.severity warning
 * @id javascript/example/empty-block
 */

import javascript


from VarAccess var, AssignExpr assign, CallExpr call, string str, string context
where str = var.getParent() + var.getLocation().toString() and
str in
["arg.sortAnimeSeasonalController.ts:68"]
and
(
    (
        call.getAnArgument().toString() = var.getParent().toString() and
        context = call.getReceiver() + "." + call.getCalleeName()
    )or
    (
        assign.getLhs().getParent().toString() = var.getParent().toString() and
        context = assign. getParent().toString()
    )
)
select  var.getParent(), var.getLocation(), context
