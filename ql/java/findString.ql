/**
 * @name find string
 * @kind problem
 * @problem.severity warning
 * @id java/example/empty-block
 */

import java

from Variable var
where var.getType() instanceof  TypeString
select var, var.getInitializer()