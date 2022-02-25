/**
 * @name find string
 * @kind problem
 * @problem.severity warning
 * @id java/example/empty-block
 */

import java

from Variable var, string namestr
where var.getType() instanceof  TypeString and namestr = var.getName().toLowerCase() and
	(namestr.regexpMatch("\\w*password\\w*") or
     namestr.regexpMatch("\\w*passwd\\w*") or
      namestr.regexpMatch("\\w*pwd\\w*") or
      namestr.regexpMatch("\\w*secret\\w*") or
      namestr.regexpMatch("\\w*token\\w*") or
      namestr.regexpMatch("\\w*auth\\w*") or
      namestr.regexpMatch("\\w*access\\w*")
	)

select var, var.getInitializer()