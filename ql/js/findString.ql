/**
 * @name find string
 * @kind problem
 * @problem.severity warning
 * @id javascript/example/empty-block
 */

import javascript

from AssignExpr assign

select assign.getTarget() , assign.getStringValue(), assign.getLocation().getStartLine(), assign.getLocation()