/**
 * @name find string
 * @kind problem
 * @problem.severity warning
 * @id csharp/example/empty-block
 */

import csharp

from AssignExpr assgin, string text, string namestr
where text = assgin.getRValue().getValue().toString() and
namestr = assgin.getLValue().(VariableAccess).getTarget().getName().toLowerCase() and
text.length() >=6 and
(namestr.regexpMatch("\\w*password\\w*") or
namestr.regexpMatch("\\w*passwd\\w*") or
namestr.regexpMatch("\\w*pwd\\w*") or
namestr.regexpMatch("\\w*secret\\w*") or
namestr.regexpMatch("\\w*token\\w*") or
namestr.regexpMatch("\\w*auth\\w*") or
namestr.regexpMatch("\\w*host\\w*") or
namestr.regexpMatch("\\w*server\\w*") or
namestr.regexpMatch("\\w*username\\w*") or
namestr.regexpMatch("\\w*account\\w*")
)
select assgin.getLValue().(VariableAccess).getTarget().getName(), text, assgin.getLocation().getStartLine(), assgin.getLocation()