/**
 * @name find string
 * @kind problem
 * @problem.severity warning
 * @id cpp/example/empty-block
 */

import cpp


from Variable var
where var.getInitializer().getExpr().getActualType().toString() = "const char *"
select var.getName().toString(), var.getInitializer().getExpr().getValue(), var.getInitializer().getLocation()

