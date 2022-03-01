/**
 * @name find string
 * @kind problem
 * @problem.severity warning
 * @id python/example/empty-scope
 */

import python

from AssignStmt var, string s
where var.getValue() instanceof Str_ and s = var.getValue().(StrConst).getText().toString() and
s.length() > 6 and s.length() < 256 and  s != "[empty string]"
select var.getATarget(), var.getValue().(StrConst).getText().toString(), var.getLocation().getStartLine(), var.getLocation()