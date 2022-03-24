/**
 * @name find string
 * @kind problem
 * @problem.severity warning
 * @id csharp/example/empty-block
 */

import csharp

from AssignExpr assgin, string text
where text = assgin.getRValue().getValue().toString()  and
text.length() >=6
select assgin.getLValue().(VariableAccess).getTarget().getName(),
assgin.getRValue().getValue(),
assgin.getLValue().(VariableAccess).getTarget().getInitializer().getLocation().getStartLine(),
assgin.getLValue().(VariableAccess).getTarget().getInitializer().getLocation()