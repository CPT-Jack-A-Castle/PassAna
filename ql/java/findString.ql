/**
 * @name find string
 * @kind problem
 * @problem.severity warning
 * @id java/example/empty-block
 */

import java

from Variable var,string text
where var.getType() instanceof  TypeString and
text = var.getInitializer().toString() and
text.length() >= 6
select var.getName().toString(), var.getInitializer().toString(), var.getInitializer().getLocation().getStartLine(), var.getInitializer().getLocation().toString()