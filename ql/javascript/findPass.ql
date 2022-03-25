/**
 * @name find string
 * @kind problem
 * @problem.severity warning
 * @id javascript/example/empty-block
 */

import javascript

from AssignExpr assign, string namestr, string contentstr

where contentstr = assign.getStringValue() and namestr = assign.getTarget().toString().toLowerCase() and
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

select namestr , contentstr, assign.getLocation().getStartLine(), assign.getLocation()