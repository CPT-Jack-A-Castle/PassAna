/**
 * @name find string
 * @kind problem
 * @problem.severity warning
 * @id java/example/empty-block
 */

import java

from Variable var
where var.getType() instanceof  TypeString

select var.getName().toString(), var.getInitializer().toString(), var.getInitializer().getLocation().getStartLine(), var.getInitializer().getLocation().toString()